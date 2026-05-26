import jax
import jax.numpy as jnp

from flax import linen as nn
from einops import rearrange
from .attentions import FMHA


def extract_patches1d(x, b):
    return rearrange(x, "batch (L_eff b) -> batch L_eff b", b=b)


def extract_patches2d(x, b):
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
    # flatten the patches
    x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
    return x


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class Embed(nn.Module):
    """Patch extraction plus linear embedding layer for 1D/2D inputs."""

    d_model: int
    b: int
    two_dimensional: bool = False

    def setup(self):
        if self.two_dimensional:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches1d

        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)

        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block with pre-norm attention and feed-forward residuals."""

    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            h=self.h,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )

        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    """Stack of transformer encoder blocks."""

    num_layers: int
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                L_eff=self.L_eff,
                transl_invariant=self.transl_invariant,
                two_dimensional=self.two_dimensional,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x


class OutputHead(nn.Module):
    """Maps encoded tokens to log-amplitudes (and phases in complex mode)."""

    d_model: int
    complex: bool = False

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm0 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm1 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x, return_z=False):

        z = self.out_layer_norm(x.sum(axis=-2))

        if return_z:
            return z

        # amp = self.norm0(self.output_layer0(z))
        amp = self.output_layer0(z)
        if self.complex:
            # sign = self.norm1(self.output_layer1(z))
            sign = self.output_layer1(z)
            out = amp + 1j * sign
        else:
            out = amp

        return jnp.sum(log_cosh(out), axis=-1)


class ViTFNQS(nn.Module):
    """Vision-transformer-style ansatz for foundational neural quantum states."""

    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    b: int
    n_coups: int
    complex: bool = False
    disorder: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        if self.disorder:
            self.patches_and_embed = Embed(
                self.d_model // 2, self.b, two_dimensional=self.two_dimensional
            )
            self.patches_and_embed_coup = Embed(
                self.d_model // 2, self.b, two_dimensional=self.two_dimensional
            )
        else:
            self.embed = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=jnp.float64,
                dtype=jnp.float64,
            )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )

        self.output = OutputHead(self.d_model, complex=self.complex)

    def __call__(self, x):
        n_coups = self.n_coups
        spins = x[..., :-n_coups]
        coups = x[..., -n_coups:]

        return_z = False
        x = jnp.atleast_2d(spins)

        if self.disorder:
            # raise NotImplementedError
            coups = jnp.broadcast_to(coups, x.shape)
            x_spins = self.patches_and_embed(x)
            x_coups = self.patches_and_embed(coups)
            x = jnp.concatenate((x_spins, x_coups), axis=-1)
        else:
            if self.two_dimensional:
                x = extract_patches2d(x, self.b)
            else:
                x = extract_patches1d(x, self.b)
            coups = jnp.broadcast_to(
                coups.reshape(*coups.shape[:-1], 1, -1), (*x.shape[:-1], n_coups)
            )
            # coups = jnp.repeat(coups[:, None], repeats=x.shape[1], axis=1)
            x = jnp.concatenate((x, coups), axis=-1)
            x = self.embed(x)

        x = self.encoder(x)

        out = self.output(x, return_z=return_z)

        return out
