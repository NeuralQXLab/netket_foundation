# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from jax.tree_util import register_pytree_node_class

from netket.hilbert import TensorHilbert

from netket.operator import DiscreteJaxOperator
from netket_foundation._src.operator.embed.base import EmbedOperator

EmbedOperator


@register_pytree_node_class
class EmbedDiscreteJaxOperator(EmbedOperator, DiscreteJaxOperator):
    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: DiscreteJaxOperator,
        subspace: int,
    ):

        if not isinstance(operator, DiscreteJaxOperator):
            raise TypeError(
                "Arguments to EmbedDiscreteJaxOperator must be "
                "subtypes of DiscreteJaxOperator. However the type is:\n\n"
                f"{type(operator)}\n"
            )
        super().__init__(hilbert, operator, subspace)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self.operator.max_conn_size

    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sub = x[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ]
        x_conn_sub, mels = self.operator.get_conn_padded(x_sub)
        x_conn = x.reshape(*x.shape[:-1], 1, x.shape[-1])
        x_conn = x_conn.repeat(x_conn_sub.shape[-2], axis=-2)
        x_conn = x_conn.at[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ].set(x_conn_sub)
        return x_conn, mels

    def to_numba_operator(self) -> "EmbedDiscreteOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from netket_foundation._src.operator.embed.discrete_operator import (
            EmbedDiscreteOperator,
        )

        return EmbedDiscreteOperator(
            self.hilbert,
            self.operator.to_numba_operator(),
            self.subspace,
        )

    def tree_flatten(self):
        data = (self.operator,)
        metadata = {"hilbert": self.hilbert, "subspace": self.subspace}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (operator,) = data
        hilbert = metadata["hilbert"]
        subspace = metadata["subspace"]

        return cls(hilbert, operator, subspace)
