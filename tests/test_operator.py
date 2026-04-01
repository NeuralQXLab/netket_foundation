"""Tests for ParametrizedOperator."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import netket_foundation as nkf
from helpers import make_hilbert, make_parameter_space, make_ising


@pytest.fixture(scope="module")
def hi():
    return make_hilbert()


@pytest.fixture(scope="module")
def ps():
    return make_parameter_space()


@pytest.fixture(scope="module")
def create_ising(hi):
    return make_ising(hi)


@pytest.fixture(scope="module")
def ham(hi, ps, create_ising):
    return nkf.operator.ParametrizedOperator(hi, ps, create_ising)


def test_hilbert_is_product(hi, ps, ham):
    """Operator's Hilbert space is the product hi * ps."""
    assert ham.hilbert == hi * ps


def test_get_conn_padded_shape(hi, ps, ham):
    """Connected states and matrix elements have compatible shapes."""
    x_phys = hi.random_state(jax.random.key(7), 8)
    x_params = ps.random_state(jax.random.key(8), 8)
    x = jnp.concatenate([x_phys, x_params], axis=-1)

    xs, mels = ham.get_conn_padded(x)

    # xs: (n_samples, max_conn, hilbert.size)  mels: (n_samples, max_conn)
    assert xs.ndim == 3
    assert mels.ndim == 2
    assert xs.shape[0] == 8
    assert mels.shape[0] == 8
    assert xs.shape[1] == mels.shape[1]
    assert xs.shape[2] == hi.size + ps.size


def test_get_conn_padded_params_preserved(hi, ps, ham):
    """Parameter portion is unchanged in all connected states."""
    x_phys = hi.random_state(jax.random.key(3), 6)
    x_params = jnp.ones((6, ps.size)) * 1.0  # fixed coupling = 1.0
    x = jnp.concatenate([x_phys, x_params], axis=-1)

    xs, _ = ham.get_conn_padded(x)

    params_in_conn = xs[..., hi.size :]
    np.testing.assert_allclose(params_in_conn, 1.0, atol=1e-6)


def test_values_match_reference_operator(hi, ps, create_ising):
    """At a fixed coupling, ParametrizedOperator agrees with the direct operator."""
    h_val = 1.0
    ref_op = create_ising(jnp.array([h_val]))

    x_phys = hi.random_state(jax.random.key(5), 10)
    x_params = jnp.full((10, ps.size), h_val)
    x_joint = jnp.concatenate([x_phys, x_params], axis=-1)

    par_op = nkf.operator.ParametrizedOperator(hi, ps, create_ising)
    xs_par, mels_par = par_op.get_conn_padded(x_joint)
    xs_ref, mels_ref = ref_op.get_conn_padded(x_phys)

    np.testing.assert_allclose(mels_par, mels_ref, atol=1e-6)
    np.testing.assert_allclose(xs_par[..., : hi.size], xs_ref, atol=1e-6)


def test_with_params_returns_operator(hi, ps, create_ising):
    """with_params(1-D array) returns a concrete operator."""
    par_op = nkf.operator.ParametrizedOperator(hi, ps, create_ising)
    op = par_op.with_params(jnp.array([1.0]))
    assert hasattr(op, "get_conn_padded")


def test_pytree_roundtrip(ham):
    """ParametrizedOperator survives JAX pytree flatten/unflatten."""
    leaves, treedef = jax.tree_util.tree_flatten(ham)
    ham2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert ham2.hilbert == ham.hilbert
