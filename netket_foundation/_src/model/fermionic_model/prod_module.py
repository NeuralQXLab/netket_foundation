from flax import linen as nn


class ProductModule(nn.Module):
    module1: nn.Module
    module2: nn.Module

    @nn.compact
    def __call__(self, x):
        return self.module1(x) + self.module2(
            x
        )  # The product of two modules is the sum in log space
