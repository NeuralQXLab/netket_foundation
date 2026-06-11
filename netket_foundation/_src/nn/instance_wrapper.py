from flax import linen as nn

import jax.numpy as jnp

from netket_foundation._src.hilbert.parameter_space import ParameterSpace


class FoundationalInstance(nn.Module):
    parameter_space: ParameterSpace
    module: nn.Module

    def setup(self):
        nn.share_scope(self, self.module)

        self.parameters = self.variable(
            "foundational",
            "parameters",
            # lambda : jnp.zeros((self.parameter_space.size,)),
            lambda: self.parameter_space.random_state(
                self.make_rng(),
            ),
        )

    # @nn.compact
    def __call__(self, x_physical):

        # Broadcast stuff to match the shape of samples along all leading axes
        parameters_bc = jnp.broadcast_to(
            self.parameters.value, x_physical.shape[:-1] + self.parameters.value.shape
        )

        # Concatenate along the last axis
        x_physical = jnp.concatenate([x_physical, parameters_bc], axis=-1)

        return self.module(x_physical)
