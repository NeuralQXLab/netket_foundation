import jax


from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.hilbert.random import random_state

from netket.utils.dispatch import dispatch


class ParameterSpace(AbstractHilbert):
    """
    Represents the vector space of a single parameter for the model.

    Behaves as a standard Hilbert space, but there are no operators defined on it.
    """

    def __init__(self, N, min=-1, max=1):
        self._N = N
        self._min = min
        self._max = max
        super().__init__()

    @property
    def size(self):
        return self._N

    @property
    def _attrs(self) -> tuple:
        return (self._N, self._min, self._max)

    def __repr__(self):
        return f"ParameterSpace(N={self._N}, min={self._min}, max={self._max})"


@dispatch
def random_state(hilb: ParameterSpace, key, batches: int, *, dtype=None):  # noqa: F811
    """If no periodic boundary conditions are present particles are positioned normally distributed around the origin.

    If periodic boundary conditions are present the particles are positioned uniformly inside the box and a small
    gaussian noise is added on top.
    If periodic boundary conditions are chosen only for certain dimensions, the periodic initialization is used for
    all of those dimensions and the free initialization is used for all the other ones.
    """
    return jax.random.uniform(
        key,
        shape=(batches, hilb.size),
        dtype=nkjax.dtype_real(dtype),
        minval=hilb._min,
        maxval=hilb._max,
    )
