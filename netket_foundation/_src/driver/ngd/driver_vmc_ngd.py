from typing import Callable, Optional
from functools import partial

import jax

from flax import core as fcore

from netket.optimizer.solver import cholesky
from netket.utils.types import Array, Optimizer, ScalarOrSchedule
from netket.operator import AbstractOperator
from netket.utils import struct, timing
from netket.vqs.mc import MCState, get_local_kernel, get_local_kernel_arguments
from netket.jax._jacobian.default_mode import JacobianMode
from netket import jax as nkjax
from netket.stats import statistics

# from netket._src.driver.abstract_optimization_driver import AbstractOptimizationDriver
from netket.driver import VMC_SR
from netket._src.ngd.sr_srt_common import get_samples_and_pdf
from netket_foundation._src.driver.ngd.sr_srt_common import sr, srt
# from netket_foundation._src.driver.ngd.srt_onthefly import srt_onthefly


class VMC_NG(VMC_SR):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR)
    with or without its kernel formulation. The two approaches lead to *exactly* the same parameter
    updates. In the kernel SR framework, the updates of the parameters can be written as:

    .. math::
        \delta \theta = \tau X(X^TX + \lambda \mathbb{I}_{2M})^{-1} f,

    where :math:`X \in R^{P \times 2M}` is the concatenation of the real and imaginary part
    of the centered Jacobian, with P the number of parameters and M the number of samples.
    The vector f is the concatenation of the real and imaginary part of the centered local
    energy. Note that, to compute the updates, it is sufficient to invert an :math:`M\times M` matrix
    instead of a :math:`P\times P` one. As a consequence, this formulation is useful
    in the typical deep learning regime where :math:`P \gg M`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.

    When `momentum` is used, this driver implements the SPRING optimizer in
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    # _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
        linear_solver: Callable[[Array, Array], Array] = cholesky,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        mode: Optional[JacobianMode] = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = False,
    ):
        r"""
        Initialize the driver.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the bare energy gradient.
            diag_shift: The diagonal shift of the curvature matrix.
            proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
            momentum: Momentum used to accumulate updates in SPRING.
            linear_solver: Callable to solve the linear problem associated to the updates of the parameters.
            mode: The mode used to compute the jacobian or vjp of the variational state.
                Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
                `real` can be used for real wavefunctions with a sign to further reduce the computational costs.
            on_the_fly: Whether to compute the QGT or NTK matrix without evaluating the full jacobian. Defaults to True.
                This ususally lowers the memory requirement and is necessary for large calculations.
            use_ntk: Whether to use the NTK instead of the QGT for the computation of the updates.
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other variational states are not supported.
            chunk_size_bwd: The chunk size to use for the backward pass (jacobian or vjp evaluation).
            collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are the linear and quadratic term in the approximation of the loss function. They are stored in the info dictionary of the driver.

        Returns:
            The new parameters, the old updates, and the info dictionary.
        """
        # self._ham = hamiltonian.collect()  # type: AbstractOperator

        # Not implemented yet
        # if not isinstance(self._ham, (ContinuousOperator, DiscreteJaxOperator)):
        #    raise TypeError("This driver only works with Jax Operators")

        super().__init__(
            hamiltonian=hamiltonian.collect(),
            optimizer=optimizer,
            diag_shift=diag_shift,
            proj_reg=proj_reg,
            momentum=momentum,
            linear_solver=linear_solver,
            variational_state=variational_state,
            chunk_size_bwd=chunk_size_bwd,
            mode=mode,
            use_ntk=use_ntk,
            on_the_fly=on_the_fly,
        )

    @property
    def update_fn(self) -> Callable:
        """Returns the function to compute the NGD update based on the evaluation mode."""
        if self.use_ntk:
            if self.on_the_fly:
                raise NotImplementedError
                # return srt_onthefly
            else:
                return srt
        else:
            if self.on_the_fly:
                raise NotImplementedError
            else:
                return sr

    @timing.timed
    def compute_loss_and_update(self):
        local_energies = self.state.local_estimators(self._ham)
        self._loss_stats = statistics(local_energies)
        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)

        samples, pdf = get_samples_and_pdf(self.state)

        self._dp, self._old_updates, self.info = self.update_fn(
            self.state._apply_fun,
            local_energies,
            self.state.parameters,
            self.state.model_state,
            samples,
            pdf=pdf,
            n_replicas=self.state.n_replicas,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd
        )

        return self._loss_stats, self._dp