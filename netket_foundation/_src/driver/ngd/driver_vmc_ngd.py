from typing import Any, Callable

import jax.numpy as jnp

from netket.optimizer.solver import cholesky_with_fallback
from netket.utils.types import Array, Optimizer, ScalarOrSchedule
from netket.operator import AbstractOperator
from netket.utils import timing, struct
from netket.vqs.mc import MCState
from netket.jax._jacobian.default_mode import JacobianMode
from netket.stats import statistics, Stats, online_statistics

# from netket._src.driver.abstract_optimization_driver import AbstractOptimizationDriver
from netket.driver import VMC_SR as NetKetVMC_SR
from netket._src.ngd.sr_srt_common import get_samples_and_pdf
from netket_foundation._src.driver.ngd.sr_srt_common import sr, srt
from netket_foundation._src.driver.ngd.srt_onthefly import srt_onthefly


class VMC_SR(NetKetVMC_SR):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and Stochastic Reconfiguration
    (SR) / Natural Gradient Descent, specialized for the foundational training scheme.

    This driver tracks :class:`netket.driver.VMC_SR`, and we refer to its documentation for
    a detailed description of the method, the available formulations (standard vs.
    kernel/minSR), the matrix-inversion solvers, and the momentum/SPRING accelerator. All of
    those options behave as documented there.

    The difference is that this driver computes the SR/NGD formulas a bit differently in order
    to better make use of the foundation training scheme: a single variational state holds
    several replicas (one per point in :class:`~netket_foundation.ParameterSpace`, e.g. a range
    of Hamiltonian parameters) and they are all trained simultaneously. The Jacobian and the
    local energies carry an extra replica dimension, which is handled explicitly so that the
    natural-gradient updates are computed per replica rather than mixing samples across
    different physical points. The progress bar and logged loss therefore report per-replica
    energy statistics (see ``log_replica_stats``) rather than a single-state energy.

    For the underlying SR/NGD derivation and references, see :class:`netket.driver.VMC_SR`.
    """

    _replica_stats: Any = struct.field(pytree_node=False, serialize=False, default=None)
    _replica_online_stats: Any = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    log_replica_stats: bool = struct.field(
        pytree_node=False, serialize=False, default=False
    )

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        proj_reg: ScalarOrSchedule | None = None,
        momentum: ScalarOrSchedule | None = None,
        linear_solver: Callable[[Array, Array], Array] = cholesky_with_fallback,
        variational_state: MCState = None,
        chunk_size_bwd: int | None = None,
        mode: JacobianMode | None = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = False,
        log_replica_stats: bool = False,
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
            on_the_fly: Whether to compute the NTK matrix without evaluating the full jacobian.
                This usually lowers the memory requirement and is necessary for large calculations.
                Only supported together with ``use_ntk=True`` (importance-sampling weights / pdf
                are not yet supported in the on-the-fly path).
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
        # "Energy" is misleading: _loss_stats reports mean energy across replicas
        # with relative (rescaled) error/variance, not a single-state energy.
        self._loss_name = "AvgEnergyReplicas"
        self.log_replica_stats = log_replica_stats

    @property
    def update_fn(self) -> Callable:
        """Returns the function to compute the NGD update based on the evaluation mode."""
        if self.use_ntk:
            if self.on_the_fly:
                return srt_onthefly
            else:
                return srt
        else:
            if self.on_the_fly:
                raise NotImplementedError(
                    "on_the_fly=True is only supported together with use_ntk=True."
                )
            else:
                return sr

    def _log_additional_data(self, log_dict):
        super()._log_additional_data(log_dict)
        if self.log_replica_stats and self._replica_stats is not None:
            for r, s in enumerate(self._replica_stats):
                log_dict[f"{self._loss_name}_r{r}"] = s

    @timing.timed
    def compute_loss_and_update(self):
        local_energies = self.state.local_estimators(self._ham)

        # local_energies has shape (n_chains, chain_length) with chains blocked
        # by replica. Compute per-replica statistics so the progress bar and
        # log reflect physically meaningful energies at each h value.
        n_replicas = self.state.n_replicas
        n_chains = local_energies.shape[0]
        chain_length = local_energies.shape[1] if local_energies.ndim > 1 else 1
        n_chains_per_replica = n_chains // n_replicas
        local_e_by_replica = local_energies.reshape(
            n_replicas, n_chains_per_replica, chain_length
        )

        decay = self._mcmc_convergence_diagnostics_ema_decay
        if decay is not None:
            if self._replica_online_stats is None:
                self._replica_online_stats = [None] * n_replicas
            self._replica_online_stats = [
                online_statistics(
                    local_e_by_replica[r],
                    old_estimator=self._replica_online_stats[r],
                    decay=decay,
                    max_lag=32,
                )
                for r in range(n_replicas)
            ]
            self._replica_stats = [
                self._replica_online_stats[r].get_stats() for r in range(n_replicas)
            ]
        else:
            self._replica_stats = [
                statistics(local_e_by_replica[r]) for r in range(n_replicas)
            ]

        # Combine per-replica Stats into a single summary.
        # Mean: average of replica means.
        # Error of mean: quadrature sum divided by n_replicas (SE of the grand mean).
        # Variance: mean of per-replica variances.
        # R_hat: max across replicas (most conservative convergence indicator).
        means = jnp.array([s.mean for s in self._replica_stats])
        errors = jnp.array([s.error_of_mean for s in self._replica_stats])
        variances = jnp.array([s.variance for s in self._replica_stats])
        rhats = jnp.array([s.R_hat for s in self._replica_stats])
        self._loss_stats = Stats(
            mean=jnp.mean(means),
            error_of_mean=jnp.sqrt(jnp.nansum(errors**2)) / n_replicas,
            variance=jnp.nansum(variances) / n_replicas,
            R_hat=jnp.nanmax(rhats),
        )

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
            chunk_size=self.chunk_size_bwd,
        )

        return self._loss_stats, self._dp
