"""Patch extraction and embedding utilities for the Fermionic ViT encoder.

The embedding stage reshapes spin configurations into spatial patches, augments
them with coupling parameters, and projects the result into the model space.
"""

from typing import Any
import jax.numpy as jnp
import jax
import netket as nk
import flax.linen as nn

def extract_patches_1d(x, graph, b):
    L = graph.n_nodes
    n_patches = L // b
    x = x.reshape(2, n_patches, b)
    x = jnp.concatenate([x[0], x[1]], axis=-1) 
    
    return x

def extract_patches2d(x, graph, b):
    """Split a spin configuration into non-overlapping square patches."""
    L_x = graph.extent[0] // b
    L_y = graph.extent[1] // b
    x = x.reshape(2, L_x, b, L_y, b)  # [spin_sub, x_patch, x_in_patch, y_patch, y_in_patch]
    x = x.transpose(0, 1, 3, 2, 4)  # [spin_sub, x_patch, y_patch, x_in_patch, y_in_patch]
    # Collapse each patch into a feature vector while preserving patch order.
    x = x.reshape(2, L_x, L_y, -1)
    x = x.reshape(2, L_x * L_y, -1)
    x = jnp.concatenate(x, axis=1)
    return x


class Embed(nn.Module):
    """Embed spin and coupling inputs into the latent patch representation."""
    d_model: int
    b: int
    n_patches: int  # given b and n_patches we can extract the system size: n_patches * (b**2) = L**2
    n_coups: int
    graph: nk.graph = None
    is_2d: bool = False
    is_equivariant: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        if self.is_2d:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches_1d

        # A single dense projection is enough once the configuration is patchified.
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )

    @nn.compact
    def __call__(self, x):
        """Project a flattened configuration and its couplings into model space."""
        n_coups = self.n_coups
        x_spin = x[..., :-n_coups]
        coups = x[..., -n_coups:]

        x = jnp.atleast_2d(x_spin)

        if self.is_2d:
            Lx = self.graph.extent[0]
            Ly = self.graph.extent[1]
            x = x.reshape(-1, 2, Lx, Ly)

            if (self.is_equivariant is True) and (self.graph is not None) and (self.b > 1):

                if self.b > 2:
                    raise NotImplementedError("Equivariance is only implemented for b>2")

                # Augment the input with translated copies to encode discrete symmetry.
                translations = []
                if self.graph.pbc[0]:
                    translations.append(jnp.roll(x, 1, axis=-2))  # x translation
                if self.graph.pbc[1]:
                    translations.append(jnp.roll(x, 1, axis=-1))  # y translation
                if self.graph.pbc[0] and self.graph.pbc[1]:
                    translations.append(jnp.roll(x, 1, axis=(-2, -1)))  # x+y translation

                x = jnp.concatenate([x] + translations, axis=0)

        else:
            L = self.graph.n_nodes
            x = x.reshape(-1, 2, L)

            if (self.is_equivariant is True) and (self.graph is not None) and (self.b > 1):
                translations = []
                if self.graph.pbc[0]:
                    for shift in range(1, self.b):
                        translations.append(jnp.roll(x, shift, axis=-1))

                x = jnp.concatenate([x] + translations, axis=0)

        # Apply patch extraction sample-wise, then broadcast the couplings to each patch.
        x = jax.vmap(self.extract_patches, in_axes=(0, None, None))(x, self.graph, self.b)
        coups = jnp.broadcast_to(
                coups.reshape(*coups.shape[:-1], 1, -1), (*x.shape[:-1], n_coups)
            )
        x = jnp.concatenate((x, coups), axis=-1)
            
        # Perform the embedding
        x = self.embed(x)
        return x