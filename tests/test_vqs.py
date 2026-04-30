"""Tests for FoundationalQuantumState."""

import pytest
import numpy as np
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
from netket.stats import Stats
from helpers import (
    make_hilbert,
    make_parameter_space,
    make_sampler,
    make_model,
    make_ising,
    make_vstate,
)


@pytest.fixture(scope="module")
def hi():
    return make_hilbert()


@pytest.fixture(scope="module")
def ps():
    return make_parameter_space()


@pytest.fixture(scope="module")
def sampler(hi):
    return make_sampler(hi)


@pytest.fixture(scope="module")
def model(ps):
    return make_model(ps)


@pytest.fixture(scope="module")
def create_ising(hi):
    return make_ising(hi)


@pytest.fixture(scope="module")
def ham(hi, ps, create_ising):
    return nkf.operator.ParametrizedOperator(hi, ps, create_ising)


@pytest.fixture(scope="module")
def vstate(sampler, model, ps):
    return make_vstate(sampler, model, ps)


def test_n_replicas(vstate):
    assert vstate.n_replicas == 4


def test_hilbert_physical(vstate, hi):
    assert vstate.hilbert_physical == hi


def test_parameter_array_shape(vstate, ps):
    pa = vstate.parameter_array
    assert pa.shape == (vstate.n_replicas, ps.size)


def test_parameter_array_assignment(vstate, ps):
    new_pa = jnp.linspace(0.9, 1.1, vstate.n_replicas).reshape(-1, ps.size)
    vstate.parameter_array = new_pa
    np.testing.assert_allclose(vstate.parameter_array, new_pa)
    # restore original
    vstate.parameter_array = jnp.linspace(0.8, 1.2, vstate.n_replicas).reshape(
        -1, ps.size
    )


def test_n_samples(vstate):
    assert vstate.n_samples == 16


def test_samples_shape(vstate, hi, ps):
    vstate.reset()
    samples = vstate.samples
    # shape: (n_chains, chain_length, hi.size + ps.size)
    assert samples.ndim == 3
    assert samples.shape[-1] == hi.size + ps.size


def test_expect_parametrized_operator(vstate, ham):
    """expect on a ParametrizedOperator returns a list of Stats, one per replica."""
    result = vstate.expect(ham)
    assert isinstance(result, list)
    assert len(result) == vstate.n_replicas
    for stats in result:
        assert isinstance(stats, Stats)
        assert np.isfinite(float(stats.mean.real))


def test_get_state_returns_mcstate(vstate):
    """get_state extracts a single-parameter MCState."""
    params = vstate.parameter_array[0]
    sub = vstate.get_state(params)
    assert sub.hilbert == vstate.hilbert_physical


def test_unsupported_sampler_raises():
    """Passing a non-Metropolis sampler raises NotImplementedError."""
    hi = make_hilbert()
    ps = make_parameter_space()
    model = make_model(ps)
    exact_sampler = nk.sampler.ExactSampler(hi)
    with pytest.raises(NotImplementedError, match="not supported"):
        nkf.FoundationalQuantumState(exact_sampler, model, ps, n_replicas=4)


def test_n_chains_not_divisible_by_n_replicas_raises():
    """n_replicas that does not divide n_chains raises ValueError."""
    hi = make_hilbert()
    ps = make_parameter_space()
    model = make_model(ps)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=4)
    with pytest.raises(ValueError, match="n_replicas"):
        nkf.FoundationalQuantumState(sampler, model, ps, n_replicas=3)
