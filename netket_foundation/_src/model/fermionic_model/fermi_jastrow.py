import flax.linen as nn
import jax.numpy as jnp
from typing import Any
from collections.abc import Callable


class foundation_fermi_Jastrow_MLP(nn.Module):
    """A small Jastrow MLP module for fermionic variational ansatzes.

    This module accepts input feature vectors that contain spin-resolved
    site occupancies followed by coupling parameters (``coups``). It
    computes a charge-like summary across spins, concatenates the
    (broadcasted) coupling values, passes the result through an MLP,
    and returns a scalar log-Jastrow value by summing over features
    and model dimensions.

    Attributes
    ----------
    n_layers: int
        Number of dense layers in the MLP.
    d_model: int
        Width of each dense layer.
    n_coups: int
        Number of coupling parameters appended to the input.
    activation: Callable
        Activation function used between layers.
    initializer: Callable
        Kernel initializer for dense layers.
    is_disordered: bool
        If True, each site is assumed to have its own coupling value.
    out_activation: Callable | None
        Optional activation applied to the MLP output before summing.
    param_dtype: Any
        Parameter dtype for Dense layers.
    """

    n_layers: int
    d_model: int
    n_coups: int
    activation: Callable = nn.gelu
    initializer: Callable = nn.initializers.lecun_normal()
    is_disordered: bool = False
    out_activation: Callable | None = None
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        """Compute the scalar log-Jastrow for input features ``x``.

        Parameters
        ----------
        x: array
            Input array with last axis containing spin-resolved site
            features followed by ``n_coups`` coupling parameters.

        Returns
        -------
        out: array
            Scalar (or batch of scalars) giving the log-Jastrow value.
        """

        # Split input into spin features and coupling parameters
        x_spin = x[..., : -self.n_coups]
        coups = x[..., -self.n_coups :]

        # Infer number of sites (assumes 2 spin channels stacked in the last dim)
        N_sites = x_spin.shape[-1] // 2

        # Reshape to (..., 2, N_sites) and sum over the spin axis to obtain
        # a charge-like feature with shape (..., 1, N_sites)
        x_charge = jnp.sum(
            x_spin.reshape(x_spin.shape[:-1] + (2, N_sites)), axis=-2, keepdims=True
        )

        if not self.is_disordered:
            # Broadcast global coupling values to all sites: (..., n_coups, N_sites)
            expanded_coups = jnp.expand_dims(coups, axis=-1)
            expanded_coups = jnp.repeat(expanded_coups, N_sites, axis=-1)
        else:
            # In the disordered case, `coups` already provides per-site values;
            # expand to shape (..., 1, N_sites) so it can be concatenated.
            expanded_coups = jnp.expand_dims(coups, axis=-2)

        # Concatenate charge and coupling channels along the feature axis:
        # resulting shape (..., num_features, N_sites)
        x_combined = jnp.concatenate([x_charge, expanded_coups], axis=-2)

        # Pass through MLP layers
        for i in range(self.n_layers):
            x_combined = nn.Dense(
                features=self.d_model,
                kernel_init=self.initializer,
                param_dtype=self.param_dtype,
                name=f"layer_{i}",
            )(x_combined)
            x_combined = self.activation(x_combined)

        # Optional output activation
        if self.out_activation is not None:
            x_combined = self.out_activation(x_combined)

        # Sum over features and model dimension to produce the scalar output
        out = jnp.sum(x_combined, axis=(-1, -2))

        return out
