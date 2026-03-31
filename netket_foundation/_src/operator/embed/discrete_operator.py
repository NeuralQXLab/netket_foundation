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

from netket.hilbert import TensorHilbert

from netket.operator import DiscreteOperator

from netket_foundation._src.operator.embed.base import EmbedOperator


class EmbedDiscreteOperator(EmbedOperator, DiscreteOperator):
    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: DiscreteOperator,
        subspace: int,
    ):
        if not isinstance(operator, DiscreteOperator):
            raise TypeError(
                "Arguments to EmbedDiscreteOperator must be "
                "subtypes of DiscreteOperator. However the type is:\n\n"
                f"{type(operator)}\n"
            )
        self._initialized = False
        super().__init__(hilbert, operator, subspace)

    def _setup(self, force: bool = False):
        if not self._initialized:
            self._initialized = True

    def get_conn_flattened(
        self,
        x: np.ndarray,
        sections: np.ndarray,
        pad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_sub = x[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ]
        x_conn_sub, mels = self.operator.get_conn_padded(x_sub)
        x_conn = x.reshape(*x.shape[:-1], 1, x.shape[-1])
        x_conn = x_conn.repeat(x_conn_sub.shape[-2], axis=-2)
        x_conn = x_conn[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ] = x_conn_sub
        return x_conn, mels

    def to_jax_operator(self) -> "EmbedDiscreteJaxOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from netket_foundation._src.operator.embed.discrete_jax_operator import (
            EmbedDiscreteJaxOperator,
        )

        return EmbedDiscreteJaxOperator(
            self.hilbert, self.operator.to_jax_operator(), self.subspace
        )
