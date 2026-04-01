"""Integration test for VMC_NG driver: single optimisation step."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket_foundation as nkf
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
def ham(hi, ps):
    return nkf.operator.ParametrizedOperator(hi, ps, make_ising(hi))


@pytest.fixture(scope="module")
def driver(sampler, model, ps, ham):
    vs = make_vstate(sampler, model, ps, seed=99)
    optimizer = optax.sgd(0.01)
    return nkf.VMC_NG(ham, optimizer, variational_state=vs, diag_shift=1e-3)


def test_single_step_runs(driver):
    """A single optimisation step completes without raising."""
    driver.run(1)


def test_energy_finite(driver):
    """After one step the per-replica energies are all finite real numbers."""
    result = driver.state.expect(driver._ham)
    for stats in result:
        e = float(stats.mean.real)
        assert np.isfinite(e), f"Energy is not finite: {e}"


def test_variables_change_after_step(driver):
    """Parameters are updated (not identical) after at least one gradient step."""
    leaves = jnp.concatenate(
        [jnp.ravel(v) for v in jax.tree_util.tree_leaves(driver.state.variables)]
    )
    assert not jnp.all(leaves == 0)
