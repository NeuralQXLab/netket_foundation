"""FMHAL encoder blocks used by the Fermionic Vision Transformer.

The encoder implements a factorized attention mechanism with optional
translation-aware parameter sharing over the patch grid.
"""

import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from einops import rearrange
from jax._src import dtypes
from typing import Any
from functools import partial

def custom_uniform(scale = 1e-2, dtype = jnp.float_):
    """Return a small uniform initializer centered at zero."""
    def init(key, shape, dtype = dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0*random.uniform(key, shape, dtype) - 1.0) * scale
    return init

def custom_id(dtype=jnp.float_):
    """Return an identity-like initializer for structured attention weights."""
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        sidex = shape[-2]
        sidey = shape[-1]
        vector = jnp.zeros(sidex * sidey, dtype=dtype)
        vector = vector.at[0].set(1.0) 
        vector = vector.reshape(sidex, sidey)
        return jnp.tile(vector, (shape[0], 1, 1))
    return init

@partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0, None, None), out_axes=1)
def roll2d(base_J, i, j, sides, graph):
    """Tile a base kernel across 2D translations to form the full attention map."""
    base_J = base_J.reshape(base_J.shape[0], sides[0], sides[1])
    if graph.pbc[0] and graph.pbc[1]:
        base_J = jnp.roll(jnp.roll(base_J, i, axis=-2), j, axis=-1)
    elif graph.pbc[0]:
        base_J = jnp.roll(base_J, i, axis=-2)
    elif graph.pbc[1]:
        base_J = jnp.roll(base_J, j, axis=-1)
    
    return base_J.reshape(base_J.shape[0], sides[0]*sides[1])

@partial(jax.vmap, in_axes=(0,None,None,None)) #vmap over the head axis
@partial(jax.vmap, in_axes = (None, 0, None, None), out_axes=1)
def roll_vertical(J: jnp.ndarray, i: jnp.ndarray, Lx:int, Ly:int):
    """Tile a base kernel along the periodic direction when only one axis is periodic."""

    J = J.reshape(Lx, Lx,Ly)
    J = jnp.roll(J, i, axis=-1)
    return J.reshape(-1, Lx*Ly)


class FactoredAttention(nn.Module):
    """Attention layer with translation-aware parameter tying."""
    n_patches: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):

        # Choose between a broad random initialization and an identity-like seed.
        if self.use_uniform_init:
            init = custom_uniform(scale=(3.0/self.n_patches)**0.5)
        else: 
            init = custom_id()

        sidex = self.graph.extent[0] // self.b
        sidey = self.graph.extent[1] // self.b

        # Build the full attention matrix from the most compact parameterization
        # compatible with the chosen symmetry setting.
        if self.is_equivariant is False:
            self.base_J = self.param("J", 
                                    init,
                                    (self.heads, self.n_patches, self.n_patches))
            self.full_J = self.base_J  # Non translational invariant.

        elif self.graph.pbc[0] != self.graph.pbc[1]:
                #we need more parameters since the translation invariance is only along the y direction

                self.base_J = self.param('J', 
                                    init, 
                                    (self.heads, sidex, self.n_patches),
                                    dtype=self.param_dtype
                                    ) #h matrices of size Lx_eff x Lx_eff x Ly_eff
                
                self.full_J = roll_vertical(self.base_J, jnp.arange(sidey), sidex, sidey) #apply the attention weights
                self.full_J = self.full_J.reshape(self.heads, self.n_patches, self.n_patches) #h matrices of size Np x Np
        
        elif self.graph.pbc[0] == self.graph.pbc[1]:

            self.base_J = self.param("J",
                                     init,
                                     (self.heads, sidex, sidey)
                                     )
            
            self.full_J = roll2d(self.base_J, jnp.arange(sidex), jnp.arange(sidey), (sidex, sidey), self.graph).reshape(self.heads, self.n_patches, self.n_patches)

        else:
            raise ValueError("You must provide either a graph for translational invariance or you must set is_translational_invariant to False")

        # Projection for Value vectors. Input x shape: (Nt, n_patches, d_in)
        # The projection maps to latent_dim; later we split into heads.
        self.v = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )
        # Final output projection.
        self.W = nn.Dense(
            self.d_model,
            kernel_init=self.initializer,
            param_dtype=self.param_dtype,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        """
        Args:
            x: input tensor of shape (Nt, n_patches, d_in)
        
        Returns:
            out: output tensor of shape (Nt, n_patches, latent_dim)
        """

        # Project values: (Nt, n_patches, latent_dim)
        v = self.v(x)

        # Reshape (Nt, n_patches, num_heads, d_head) -> (Nt, num_heads, n_patches, d_head)
        v = rearrange(v, "Nt np (h d) -> Nt np h d", h=self.heads)
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_out = jnp.einsum("hij,thid->thjd", self.full_J, v)
        # attn_out is shape (Nt, num_heads, n_patches, d_head)

        # Concatenate heads: first transpose to (Nt, n_patches, num_heads, d_head) then reshape.
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = rearrange(attn_out, "Nt np h d -> Nt np (h d)")

        out = self.W(attn_out)
        return out, self.full_J

class EncoderBlock(nn.Module):
    """Pre-norm residual transformer block built on the factorized attention."""
    n_patches: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        self.attn = FactoredAttention(n_patches=self.n_patches, 
                                        d_model=self.d_model, 
                                        heads=self.heads, 
                                        b=self.b,
                                        graph=self.graph, 
                                        is_equivariant=self.is_equivariant,
                                        use_uniform_init = self.use_uniform_init,
                                        param_dtype=self.param_dtype, 
                                        initializer=self.initializer
                                        )
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
        # Feed forward layer
        self.ff = nn.Sequential(
            [
                nn.Dense(
                    2 * self.d_model,
                    kernel_init=self.initializer,
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=self.initializer,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):

        # Standard residual pattern: attention update followed by feed-forward update.
        x_att, att = self.attn(self.layer_norm_1(x))
        x = x + x_att
        x = x + self.ff(self.layer_norm_2(x))
        return x, att


class Encoder_FMHAL_roll2d(nn.Module):
    """Stack of FMHAL encoder blocks with attention-map inspection support."""
    n_patches: int
    n_layers: int
    d_model: int
    heads: int
    b: int
    graph: nk.graph = None
    is_equivariant: bool = False
    use_uniform_init: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        # Materialize a homogeneous stack of encoder blocks.
        self.layers = [
            EncoderBlock(n_patches=self.n_patches, 
                         d_model=self.d_model, 
                         b=self.b,
                         heads=self.heads, 
                         graph=self.graph, 
                         is_equivariant=self.is_equivariant,
                         use_uniform_init = self.use_uniform_init,
                         param_dtype=self.param_dtype, 
                         initializer=self.initializer
                         )

            for _ in range(self.n_layers)
        ]

    def __call__(self, x):

        # Each layer returns both the transformed representation and the attention map.
        for _, l in enumerate(self.layers):
            x = l(x)[0]

        return x

    def get_attention(self, x):
        """Return the per-layer attention maps for a single forward pass."""
        attention_maps = []
        for l in self.layers:
            _, attn_map = l(x)
            attention_maps.append(attn_map)
            x, _ = l(x)
        return attention_maps
