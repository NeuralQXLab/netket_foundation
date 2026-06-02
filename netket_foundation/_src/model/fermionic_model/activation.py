"""Activation utilities for fermionic models.

This module provides specialized activations used in the foundational
fermionic models. The implementations are chosen to be numerically
stable and to support complex-valued inputs where appropriate.
"""

import jax.numpy as jnp


def log_cosh(x):
    """Numerically stable log-cosh activation.

    This implementation is safe for large-magnitude inputs and supports
    complex-valued tensors by operating on the real part for sign
    stabilization. It implements:

        log(cosh(x)) = x + log1p(exp(-2*x)) - log(2)

    Parameters
    ----------
    x : array_like
        Input array (real or complex).

    Returns
    -------
    array_like
        Elementwise log-cosh of the input.
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)
