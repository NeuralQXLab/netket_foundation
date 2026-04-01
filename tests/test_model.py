"""Tests for ViTFNQS forward pass."""

import pytest
import jax
import jax.numpy as jnp
from netket_foundation._src.model.vit import ViTFNQS

# System layout: 4 spins, patch size 2 → L_eff=2; 1 coupling
L = 4
B = 2
L_EFF = L // B
N_COUPS = 1


def _make_model(**kwargs):
    defaults = dict(
        num_layers=1,
        d_model=4,
        heads=2,
        L_eff=L_EFF,
        n_coups=N_COUPS,
        b=B,
        complex=False,
        disorder=False,
        transl_invariant=False,
        two_dimensional=False,
    )
    defaults.update(kwargs)
    return ViTFNQS(**defaults)


def _random_input(batch, key=None):
    """Joint input: (batch, L + N_COUPS)."""
    if key is None:
        key = jax.random.key(0)
    spins = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(batch, L))
    coups = jnp.ones((batch, N_COUPS)) * 1.0
    return jnp.concatenate([spins, coups], axis=-1).astype(jnp.float64)


def test_output_shape():
    model = _make_model()
    x = _random_input(8)
    params = model.init(jax.random.key(1), x)
    y = model.apply(params, x)
    assert y.shape == (8,)


def test_output_is_real():
    model = _make_model(complex=False)
    x = _random_input(6)
    params = model.init(jax.random.key(2), x)
    y = model.apply(params, x)
    assert jnp.isrealobj(y)


def test_output_is_complex():
    model = _make_model(complex=True)
    x = _random_input(6)
    params = model.init(jax.random.key(3), x)
    y = model.apply(params, x)
    assert jnp.iscomplexobj(y)


def test_no_nan_or_inf():
    model = _make_model()
    x = _random_input(16)
    params = model.init(jax.random.key(4), x)
    y = model.apply(params, x)
    assert jnp.all(jnp.isfinite(y))


def test_batch_size_one():
    model = _make_model()
    x = _random_input(1)
    params = model.init(jax.random.key(5), x)
    y = model.apply(params, x)
    assert y.shape == (1,)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_depth_variants(num_layers):
    model = _make_model(num_layers=num_layers)
    x = _random_input(4)
    params = model.init(jax.random.key(6), x)
    y = model.apply(params, x)
    assert y.shape == (4,)
    assert jnp.all(jnp.isfinite(y))
