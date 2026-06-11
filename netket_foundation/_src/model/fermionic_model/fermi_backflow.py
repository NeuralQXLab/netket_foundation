from functools import partial
from typing import Any
import jax.numpy as jnp
import flax.linen as nn
import jax
from netket.jax import logsumexp_cplx

"""Backflow and generalized backflow modules with Jastrow support.

This module implements two related backflow-based Slater form
constructors used in the foundational fermionic models:

- `foundation_backflow`: a backflow with a translationally invariant
  Jastrow-like addition to a mean-field orbital matrix, split into
  separate UP/DOWN determinants.
- `foundation_generalized_backflow`: a generalized version where a
  single determinant over the mixed spin-orbital basis is formed.

Both classes accept a `model` callable that produces additive
corrections to a background mean-field matrix `M` and return log
Slater determinant amplitudes.
"""

default_init_func = nn.initializers.lecun_normal()


def _log_det(A):
    """Return log-determinant with sign as a complex value.

    The function returns log(|det(A)|) + i*arg(sign), encoded so the
    result is a complex number suitable for use with complex-valued
    log-amplitudes.
    """
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


_log_det = jax.jit(_log_det)


class foundation_backflow(nn.Module):
    """Backflow module that forms separate UP/DOWN Slater determinants.

    The module builds a mean-field matrix `M` (optionally initialized to
    a Fermi sea), adds the (reshaped) output of `model` as a correction,
    extracts submatrices for UP and DOWN fermions, and returns the sum
    of the two log-determinants.
    """

    model: Any
    hilbert: Any  # SpinOrbitalFermions
    # Lattice used to compute the distance between sites for the jastrow.
    # If none, it is not used.
    graph: Any = None
    enforce_spin_flip: bool = False
    mean_field_init: str = "default"
    param_dtype: Any = float
    initializer: Any = nn.initializers.he_uniform()

    def fermi_sea(self, nf_i):
        """Initialize the visible orbitals with the non-interacting Fermi sea.

        The function diagonalizes a nearest-neighbour tight-binding model on
        `self.graph` and returns the lowest `nf_i` eigenvectors for each
        spin channel concatenated together.
        """

        HFS = jnp.zeros(
            (self.hilbert.n_orbitals, self.hilbert.n_orbitals), dtype=self.param_dtype
        )

        for i, j in self.graph.edges():
            HFS = HFS.at[i, j].set(-1.0)
            HFS = HFS.at[j, i].set(-1.0)

        _, eigvecs = jnp.linalg.eigh(HFS)
        mf_per_spin = eigvecs[:, :nf_i]
        return jnp.concatenate([mf_per_spin, mf_per_spin], axis=-1)

    def batch_spin_flip(self, n):
        """Augment a batch with spin-flipped copies preserving trailing params.

        The input `n` is expected to contain concatenated `n_up`, `n_down`,
        and trailing coupling parameters. The flipped batch swaps the up
        and down occupations and reattaches the same coupling parameters.
        """

        if self.hilbert.n_fermions_per_spin[0] != self.hilbert.n_fermions_per_spin[1]:
            raise ValueError(
                "Spin sectors must have the same number of fermions in order to enforce spin-flip symmetry."
            )

        N_orb = self.hilbert.n_orbitals

        # Split input into up, down and coupling parameters
        n_up = n[:, :N_orb]
        n_down = n[:, N_orb : 2 * N_orb]
        couplings = n[:, 2 * N_orb :]

        # Build flipped samples and concatenate to the original batch
        n_flip = jnp.concatenate([n_down, n_up, couplings], axis=1)
        n_combined = jnp.concatenate([n, n_flip], axis=0)

        return n_combined

    def psi_eval_spin_flip(self, logpsi):
        """Project log-amplitudes onto the spin-flip symmetric subspace.

        Assumes the batch contains pairs (original, spin-flipped) so that
        `logpsi` can be reshaped to shape (Nbatch, 2) and combined using
        a complex log-sum-exp.
        """

        logpsi_flipped = logpsi.reshape((-1, 2), order="F")
        logpsi_symm = logsumexp_cplx(a=logpsi_flipped, b=1 / 2, axis=1)
        return logpsi_symm

    @nn.compact
    def __call__(self, n):

        # Optionally augment batch with spin-flipped samples
        if self.enforce_spin_flip:
            n = self.batch_spin_flip(n)

        # Evaluate the corrective model (e.g. Transformer/backflow network)
        F = self.model(n)

        @partial(jnp.vectorize, signature="(n),(m)->()")
        def log_sdj(n_vec, F_vec):

            N_orb = self.hilbert.n_orbitals

            if self.mean_field_init == "default":
                M = self.param(
                    "M",
                    self.initializer,
                    (N_orb, self.hilbert.n_fermions),
                    self.param_dtype,
                )

            elif self.mean_field_init == "fermi_sea":
                M = self.param(
                    "M",
                    lambda *args: self.fermi_sea(self.hilbert.n_fermions_per_spin[0]),
                    (N_orb, self.hilbert.n_fermions),
                    self.param_dtype,
                )

            # Determine occupied orbital indices for up and down spins
            R_u = n_vec[:N_orb].nonzero(size=self.hilbert.n_fermions_per_spin[0])[0]
            R_d = n_vec[N_orb : 2 * N_orb].nonzero(
                size=self.hilbert.n_fermions_per_spin[1]
            )[0]

            # Add the network output (reshaped) to the background matrix M
            M += F_vec.reshape(M.shape)

            # Extract submatrices for UP and DOWN and compute log-dets
            A_u = M[:, : self.hilbert.n_fermions_per_spin[0]][R_u]
            A_d = M[:, self.hilbert.n_fermions_per_spin[0] :][R_d]

            return _log_det(A_u) + _log_det(A_d)

        log_slater = log_sdj(n, F)

        # If requested, project onto spin-flip symmetric subspace
        if self.enforce_spin_flip:
            log_slater = self.psi_eval_spin_flip(log_slater)

        return log_slater


class foundation_generalized_backflow(nn.Module):
    model: Any
    hilbert: Any  # SpinOrbitalFermions
    enforce_spin_flip: bool = False
    param_dtype: Any = float
    initializer: Any = nn.initializers.he_uniform()

    def batch_spin_flip(self, n):

        if self.hilbert.n_fermions_per_spin[0] != self.hilbert.n_fermions_per_spin[1]:
            raise ValueError(
                "Spin sectors must have the same number of fermions in order to enforce spin-flip symmetry."
            )

        N_orb = self.hilbert.n_orbitals
        n_up = n[:, :N_orb]
        n_down = n[:, N_orb : 2 * N_orb]
        couplings = n[:, 2 * N_orb :]

        n_flip = jnp.concatenate([n_down, n_up, couplings], axis=1)
        n_combined = jnp.concatenate([n, n_flip], axis=0)

        return n_combined

    def psi_eval_spin_flip(self, logpsi):

        logpsi_flipped = logpsi.reshape((-1, 2), order="F")
        logpsi_symm = logsumexp_cplx(a=logpsi_flipped, b=1 / 2, axis=1)
        return logpsi_symm

    @nn.compact
    def __call__(self, n):

        if self.enforce_spin_flip:
            n = self.batch_spin_flip(n)

        F = self.model(n)

        @partial(jnp.vectorize, signature="(n),(m)->()")
        def log_sdj(n_vec, F_vec):

            N_orb = self.hilbert.n_orbitals
            N_fermions = self.hilbert.n_fermions

            # The background matrix M spans the full spin-orbital space
            M = self.param(
                "M",
                self.initializer,
                (2 * N_orb, N_fermions),
                self.param_dtype,
            )

            # Find indices of all occupied spin-orbitals
            R_mixed = n_vec[: 2 * N_orb].nonzero(size=N_fermions)[0]

            # Add network correction and extract the mixed submatrix
            M += F_vec.reshape(M.shape)
            A_mixed = M[R_mixed, :]

            # Return log-determinant of the single mixed Slater matrix
            return _log_det(A_mixed)

        log_slater = log_sdj(n, F)

        if self.enforce_spin_flip:
            log_slater = self.psi_eval_spin_flip(log_slater)

        return log_slater
