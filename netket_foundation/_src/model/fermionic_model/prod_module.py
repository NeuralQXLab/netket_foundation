"""Utility module to combine two submodules in log-space.

`ProductModule` composes two modules by summing their outputs. This is
useful when the two modules produce log-amplitudes (e.g. backflow +
Jastrow) and the combined log-amplitude is the sum of the two.
"""

from flax import linen as nn


class ProductModule(nn.Module):
    """Sum outputs of two modules (suitable when outputs are log-amplitudes)."""

    module1: nn.Module
    module2: nn.Module

    @nn.compact
    def __call__(self, x):
        # When modules return log-amplitudes, the combined log-amplitude is
        # the sum of the individual log-amplitudes.
        return self.module1(x) + self.module2(x)
