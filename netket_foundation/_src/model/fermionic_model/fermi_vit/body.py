"""Top-level Fermionic Vision Transformer model assembly.

This module connects the patch embedding, FMHAL encoder, and readout head into
one Flax module that maps particle configurations and couplings to the final
wavefunction output.
"""

import flax.linen as nn
from typing import Any
import jax.numpy as jnp
from functools import partial
import netket as nk

from .embed import Embed
from .fmhal1d import Encoder_FMHAL_roll1d
from .fmhal2d import Encoder_FMHAL_roll2d
from .output import OuputHead


class foundation_ViT_trans_equi(nn.Module):
    """End-to-end Fermionic ViT model with optional equivariance and invariance."""

    n_layers: int
    d_model: int
    d_output: int
    d_latent: int
    heads: int
    b: int
    n_patches: int
    n_coups: int
    graph: nk.graph
    is_2d: bool = False
    is_equivariant: bool = False
    make_it_invariant: bool = False
    use_uniform_init: bool = False
    out_activation: Any = nn.tanh
    complex: bool = False
    param_dtype: Any = jnp.float64
    initializer: Any = nn.initializers.lecun_normal()

    def setup(self):
        # The model is intentionally assembled from small, testable submodules.
        self.embedding = Embed(
            d_model=self.d_model,
            b=self.b,
            n_patches=self.n_patches,
            graph=self.graph,
            n_coups=self.n_coups,
            is_equivariant=self.is_equivariant,
            param_dtype=self.param_dtype,
            initializer=self.initializer,
        )

        if self.is_2d:
            self.encoder = Encoder_FMHAL_roll2d(
                n_patches=self.n_patches,
                b=self.b,
                graph=self.graph,
                n_layers=self.n_layers,
                d_model=self.d_model,
                use_uniform_init=self.use_uniform_init,
                heads=self.heads,
                is_equivariant=self.is_equivariant,
                initializer=self.initializer,
                param_dtype=self.param_dtype,
            )
        else:
            self.encoder = Encoder_FMHAL_roll1d(
                n_patches=self.n_patches,
                b=self.b,
                graph=self.graph,
                n_layers=self.n_layers,
                d_model=self.d_model,
                use_uniform_init=self.use_uniform_init,
                heads=self.heads,
                is_equivariant=self.is_equivariant,
                initializer=self.initializer,
                param_dtype=self.param_dtype,
            )

        self.out = OuputHead(
            d_model=self.d_model,
            d_latent=self.d_latent,
            d_output=self.d_output,
            out_activation=self.out_activation,
            make_it_invariant=self.make_it_invariant,
            is_equivariant=self.is_equivariant,
            param_dtype=self.param_dtype,
            is_complex=self.complex,
            initializer=self.initializer,
        )

    @nn.compact
    def __call__(self, x):
        """Apply the model independently to each configuration in the batch."""

        # Preserve the leading batch dimensions and flatten only the particle axis.
        d_shape_in = x.shape[-1]  # batch shape
        batch_shape_in = x.shape[:-1]
        x = x.reshape(-1, d_shape_in)

        # Vectorize the single-sample wavefunction evaluation across the batch.
        @partial(jnp.vectorize, signature="(x)->(n)")
        def compute_wavefunc(x):
            """Compute the model output for a single flattened configuration.

            The function is deliberately written for use with ``jnp.vectorize``
            so that the module can be applied efficiently across an arbitrary
            leading batch shape after flattening the particle axis.
            """
            x = self.embedding(x)
            x = self.encoder(x)
            return self.out(x)

        out = compute_wavefunc(x)
        out = out.reshape(*batch_shape_in, -1)

        return out
