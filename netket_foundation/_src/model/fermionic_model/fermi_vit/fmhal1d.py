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
    def init(key, shape, dtype = dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return (2.0*random.uniform(key, shape, dtype) - 1.0) * scale
    return init

def custom_id(dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        
        # CASO 1: Equivariante (shape richiesta è 2D: heads, n_patches)
        if len(shape) == 2:
            heads, n_patches = shape
            vector = jnp.zeros(n_patches, dtype=dtype)
            vector = vector.at[0].set(1.0) 
            return jnp.tile(vector, (heads, 1))
            
        # CASO 2: Non Equivariante (shape richiesta è 3D: heads, n_patches, n_patches)
        elif len(shape) == 3:
            heads, n_patches, _ = shape
            # Per il caso normale, l'identità è la classica matrice diagonale jnp.eye!
            identity_matrix = jnp.eye(n_patches, dtype=dtype)
            return jnp.tile(identity_matrix, (heads, 1, 1))
            
        else:
            raise ValueError(f"Shape non supportata per custom_id: {shape}")
            
    return init

# in_axes=(None, 0) significa: mantieni fisso base_J, fai scorrere i.
# out_axes=1 significa: impila i risultati del vmap lungo l'asse 1 (diventano le righe della matrice).
@partial(jax.vmap, in_axes=(None, 0), out_axes=1)
def roll1d(base_J, i):
    # base_J ha dimensione (heads, n_patches)
    # jnp.roll fa slittare l'ultimo asse (le patch) di 'i' posizioni
    return jnp.roll(base_J, i, axis=-1)

class FactoredAttention(nn.Module):
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

        if self.use_uniform_init:
            init = custom_uniform(scale=(3.0/self.n_patches)**0.5)
        else: 
            init = custom_id()

        if self.is_equivariant is False:
            self.base_J = self.param("J", 
                                    init,
                                    (self.heads, self.n_patches, self.n_patches))
            self.full_J = self.base_J  # Non translational invariant.

        elif self.graph.pbc[0]:
                #we need more parameters since the translation invariance is only along the y direction

                self.base_J = self.param('J', 
                                    init, 
                                    (self.heads, self.n_patches),
                                    dtype=self.param_dtype
                                    )
                
                self.full_J = roll1d(self.base_J, jnp.arange(self.n_patches))
                self.full_J = self.full_J.reshape(self.heads, self.n_patches, self.n_patches) #h matrices of size Np x Np
                # jax.debug.print("🤯 {x} 🤯", x=self.full_J)
        
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

        # Apply attention per head. (Einstein summation multiplies J and v along patch dimension.)
        attn_out = jnp.einsum("hij,thid->thjd", self.full_J, v)
        # attn_out is shape (Nt, num_heads, n_patches, d_head)

        # Concatenate heads: first transpose to (Nt, n_patches, num_heads, d_head) then reshape.
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = rearrange(attn_out, "Nt np h d -> Nt np (h d)")

        out = self.W(attn_out)
        return out, self.full_J

class EncoderBlock(nn.Module):
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

        x_att, att = self.attn(self.layer_norm_1(x))
        x = x + x_att
        x = x + self.ff(self.layer_norm_2(x))
        return x, att


class Encoder_FMHAL_roll1d(nn.Module):
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
        for _, l in enumerate(self.layers):
            x = l(x)[0]

        return x

    def get_attention(self, x):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l(x)
            attention_maps.append(attn_map)
            x, _ = l(x)
        return attention_maps
