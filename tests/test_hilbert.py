"""Tests for ParameterSpace (Hilbert space for model parameters)."""

import pytest
import numpy as np
import jax
import netket_foundation as nkf


@pytest.mark.parametrize("N", [1, 3])
def test_size(N):
    ps = nkf.ParameterSpace(N=N, min=-1, max=1)
    assert ps.size == N


@pytest.mark.parametrize(
    "lo, hi",
    [(-1.0, 1.0), (0.8, 1.2), (0.0, 5.0)],
)
def test_random_state_within_bounds(lo, hi):
    ps = nkf.ParameterSpace(N=3, min=lo, max=hi)
    samples = ps.random_state(jax.random.key(42), 200)
    assert samples.shape == (200, 3)
    assert float(samples.min()) >= lo
    assert float(samples.max()) <= hi


def test_random_state_reproducible():
    ps = nkf.ParameterSpace(N=2, min=0.0, max=1.0)
    s1 = ps.random_state(jax.random.key(0), 10)
    s2 = ps.random_state(jax.random.key(0), 10)
    np.testing.assert_array_equal(s1, s2)


def test_random_state_different_seeds():
    ps = nkf.ParameterSpace(N=2, min=0.0, max=1.0)
    s1 = ps.random_state(jax.random.key(0), 10)
    s2 = ps.random_state(jax.random.key(1), 10)
    assert not np.allclose(s1, s2)


def test_repr():
    ps = nkf.ParameterSpace(N=2, min=-1, max=1)
    r = repr(ps)
    assert "ParameterSpace" in r
    assert "N=2" in r


def test_equality():
    ps1 = nkf.ParameterSpace(N=1, min=0.0, max=1.0)
    ps2 = nkf.ParameterSpace(N=1, min=0.0, max=1.0)
    ps3 = nkf.ParameterSpace(N=2, min=0.0, max=1.0)
    assert ps1 == ps2
    assert ps1 != ps3
