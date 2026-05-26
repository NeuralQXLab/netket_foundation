import jax.numpy as jnp


def log_cosh(x):
    """Numerically stable log-cosh activation used by the readout head."""
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)
