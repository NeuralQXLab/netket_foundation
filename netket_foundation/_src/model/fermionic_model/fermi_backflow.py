from functools import partial
from typing import Any
import jax.numpy as jnp
import flax.linen as nn
import jax
from netket.jax import logsumexp_cplx

default_init_func = nn.initializers.lecun_normal()


def _log_det(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


_log_det = jax.jit(_log_det)


class foundation_backflow(nn.Module):
    """
    Backflow with a jastrow factor that is only distance-dependent (aka, translational invariant).
    """

    model: Any
    hilbert: Any  # SpinOrbitalFermions
    """
    Lattice used to compute the distance between sites for the jastrow.
    If none, it is not used.
    """
    graph: Any = None
    enforce_spin_flip: bool = False
    mean_field_init: str = "default"
    param_dtype: Any = float
    initializer: Any = nn.initializers.he_uniform()

    # orbital init functions
    def fermi_sea(self, nf_i):
        """initializes the visible orbitals as the Fermi sea.
        This is done by diagonalizing the tight-binding model."""

        HFS = jnp.zeros(
            (self.hilbert.n_orbitals, self.hilbert.n_orbitals), dtype=self.param_dtype
        )

        for i, j in self.graph.edges():
            HFS = HFS.at[i, j].set(-1.0)
            HFS = HFS.at[j, i].set(-1.0)

        _, eigvecs = jnp.linalg.eigh(HFS)  # diagonalize the single-particle Hamiltonian
        mf_per_spin = eigvecs[:, :nf_i]
        return jnp.concatenate([mf_per_spin, mf_per_spin], axis=-1)

    def batch_spin_flip(self, n):
        """From a batch of samples, include the spin-flipped states.
        COMPATIBLE WITH FOUNDATION MODELS (preserves trailing parameters).
        """

        if self.hilbert.n_fermions_per_spin[0] != self.hilbert.n_fermions_per_spin[1]:
            raise ValueError(
                "Spin sectors must have the same number of fermions in order to enforce spin-flip symmetry."
            )

        N_orb = self.hilbert.n_orbitals

        # Dividiamo esattamente l'array nelle sue 3 componenti fisiche
        n_up = n[:, :N_orb]
        n_down = n[:, N_orb : 2 * N_orb]
        couplings = n[
            :, 2 * N_orb :
        ]  # Parametri foundation (U, disordine, ecc.). Se vuoto, JAX lo gestisce senza errori.

        # Scambiamo solo UP e DOWN, e ri-attacchiamo i parametri foundation identici
        n_flip = jnp.concatenate([n_down, n_up, couplings], axis=1)

        # Concateniamo il batch originale con il batch flippato
        n_combined = jnp.concatenate([n, n_flip], axis=0)

        return n_combined

    def psi_eval_spin_flip(self, logpsi):
        """From a batch of log-amplitudes that include the spin-flipped states,
        return the log amplitudes projected onto the spin-flip symmetric subspace."""

        logpsi_flipped = logpsi.reshape(
            (-1, 2), order="F"
        )  # reshape to (Nbatch, Nsymm=2) [log(psi(sigma)), log(psi(Psigma))]
        logpsi_symm = logsumexp_cplx(a=logpsi_flipped, b=1 / 2, axis=1)
        return logpsi_symm

    @nn.compact
    def __call__(self, n):

        # spin flipped samples
        if self.enforce_spin_flip:
            n = self.batch_spin_flip(n)

        # Passiamo al modello (il Transformer) l'array intero (spin + couplings)
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

            # ATTENZIONE: CORREZIONE FOUNDATION QUI!
            # Prima R_d prendeva tutto da N_orb alla fine. Ora limitiamo esattamente al blocco DOWN.
            R_u = n_vec[:N_orb].nonzero(size=self.hilbert.n_fermions_per_spin[0])[0]
            R_d = n_vec[N_orb : 2 * N_orb].nonzero(
                size=self.hilbert.n_fermions_per_spin[1]
            )[0]

            # reshape into M and add
            M += F_vec.reshape(M.shape)

            # Estraiamo le sottomatrici (una per UP, una per DOWN)
            A_u = M[:, : self.hilbert.n_fermions_per_spin[0]][R_u]
            A_d = M[:, self.hilbert.n_fermions_per_spin[0] :][R_d]

            return _log_det(A_u) + _log_det(A_d)

        log_slater = log_sdj(n, F)

        # project on spin flip subspace
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
            N_fermions = (
                self.hilbert.n_fermions
            )  # Numero TOTALE di fermioni (UP + DOWN)

            # 1. La matrice M ora copre l'intero spazio degli spin-orbitali (2 * N_orb)
            M = self.param(
                "M",
                self.initializer,
                (2 * N_orb, N_fermions),
                self.param_dtype,
            )

            # 2. Troviamo gli indici di tutti i fermioni in un colpo solo.
            # Gli spin DOWN avranno automaticamente indici da N_orb a 2*N_orb - 1
            R_mixed = n_vec[: 2 * N_orb].nonzero(size=N_fermions)[0]

            # 3. Sommiamo l'output della rete al background.
            M += F_vec.reshape(M.shape)

            # 4. Estraiamo la sottomatrice unica di dimensione (N_fermions x N_fermions)
            A_mixed = M[R_mixed, :]

            # 5. Restituiamo il logaritmo di un UNICO determinante
            return _log_det(A_mixed)

        log_slater = log_sdj(n, F)

        if self.enforce_spin_flip:
            log_slater = self.psi_eval_spin_flip(log_slater)

        return log_slater
