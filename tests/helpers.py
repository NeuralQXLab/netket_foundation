"""Shared building blocks for netket_foundation tests.

Test files import these helpers to define their own local fixtures,
keeping each file self-contained while avoiding code duplication.
"""

import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
from netket_foundation._src.model.vit import ViTFNQS

# Small system: 4 spins, patch size 2 → L_eff = 2
L = 4
B = 2
L_EFF = L // B


def make_hilbert():
    return nk.hilbert.Spin(0.5, L)


def make_parameter_space():
    return nkf.ParameterSpace(N=1, min=0.8, max=1.2)


def make_sampler(hi):
    return nk.sampler.MetropolisLocal(hi, n_chains=4)


def make_model(ps):
    return ViTFNQS(
        num_layers=1,
        d_model=4,
        heads=2,
        L_eff=L_EFF,
        n_coups=ps.size,
        b=B,
        complex=False,
        disorder=False,
        transl_invariant=False,
        two_dimensional=False,
    )


def make_ising(hi):
    """Return a function params -> transverse-field Ising Hamiltonian."""

    def create_operator(params):
        h = params[0]
        ha_x = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
        ha_zz = sum(
            nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
            for i in range(hi.size)
        )
        return -h * ha_x - ha_zz

    return create_operator


def make_vstate(sampler, model, ps, *, seed=0):
    vs = nkf.FoundationalQuantumState(
        sampler, model, ps, n_samples=16, n_replicas=4, seed=seed
    )
    vs.parameter_array = jnp.linspace(0.8, 1.2, vs.n_replicas).reshape(-1, 1)
    return vs
