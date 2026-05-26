"""Readout head for the Fermionic Vision Transformer."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class OuputHead(nn.Module):
    """Project encoder features to the final real or complex wavefunction output."""

    d_model: int
    d_latent: int
    d_output: int
    out_activation: Any
    make_it_invariant: bool = False
    is_equivariant: bool = False
    param_dtype: Any = jnp.float64
    is_complex: bool = False
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):

        # Shared normalization and expansion layers used by all output modes.
        self.layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.dense_expand = nn.Dense(
            self.d_latent,
            param_dtype=self.param_dtype,
            kernel_init=self.initializer,
            bias_init=jax.nn.initializers.zeros,
        )

        if self.is_complex:
            # Complex outputs are represented by independent amplitude and sign heads.
            self.norm_amp = nn.LayerNorm(
                param_dtype=self.param_dtype,
            )
            self.norm_sign = nn.LayerNorm(param_dtype=self.param_dtype)

            self.output_layer_amp = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype,
                kernel_init=self.initializer,
                bias_init=jax.nn.initializers.zeros,
            )

            self.output_layer_sign = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype,
                kernel_init=self.initializer,
                bias_init=jax.nn.initializers.zeros,
            )

        else:
            self.output_layer = nn.Dense(
                self.d_output,
                param_dtype=self.param_dtype,
                kernel_init=self.initializer,
                bias_init=jax.nn.initializers.zeros,
            )

    def __call__(self, x):

        if self.make_it_invariant:

            # Pool translation copies before the final projection when invariance is requested.
            x = self.layer_norm(x.sum(axis=-2))  # (N_T, N_P, d) -> (N_T, d)
            x = self.dense_expand(x)  # (N_T, d) -> (N_T, N_lat)
            x = nn.gelu(x)
            x = self.out_layer_norm(
                x.sum(axis=-2)
            )  # Pool over the translation dimension (N_T, N_lat) -> (N_lat)
            x = x.flatten()

            if self.is_complex:

                # Keep amplitude and sign channels separate so each can be normalized independently.
                amp = self.norm_amp(self.output_layer_amp(x))
                sign = self.norm_sign(self.output_layer_sign(x))
                out = amp + 1j * sign

            else:

                out = self.output_layer(x)

        elif self.is_equivariant:

            # Preserve the translation axis and project each orbit element independently.

            x = x.reshape(-1, self.d_model)  # (N_P, N_T, d) -> (N_o, d)
            x = self.dense_expand(x)  # (N_o, d) -> (N_o, N_lat)
            x = nn.gelu(x)
            out = self.out_layer_norm(
                self.output_layer(x)
            ).flatten()  # (N_o, N_lat) -> (N_o, N_fermions)
        else:

            # In the non-equivariant branch, aggregate patch features before the readout.

            x = self.dense_expand(x)
            x = nn.gelu(x)
            x = self.layer_norm(x.sum(axis=-2))  # (N_P, d) -> (d)
            out = self.out_layer_norm(self.output_layer(x.flatten()))

        return self.out_activation(out)
