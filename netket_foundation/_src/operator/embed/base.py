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


from abc import ABC


from netket.hilbert import TensorHilbert

from netket.operator._abstract_operator import AbstractOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator


class EmbedOperator(ABC):
    def __new__(cls, hi, op, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `SumOperator(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from netket_foundation._src.operator.embed.operator import (
            EmbedGenericOperator,
        )

        # from .discrete_operator import SumDiscreteOperator
        from netket_foundation._src.operator.embed.discrete_jax_operator import (
            EmbedDiscreteJaxOperator,
        )

        # from .continuous import SumContinuousOperator

        if cls is EmbedOperator:
            if isinstance(op, DiscreteJaxOperator):
                cls = EmbedDiscreteJaxOperator
            # elif isinstance(op, DiscreteOperator):
            #     cls = SumDiscreteOperator
            # elif isinstance(op, ContinuousOperator):
            #     cls = SumContinuousOperator
            else:
                cls = EmbedGenericOperator
        return super().__new__(cls)

    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: AbstractOperator,
        subspace: int,
        **kwargs,
    ):
        r"""Constructs a Sum of Operators.

        Args:
            *hil
            b: An iterable object containing at least 1 hilbert space.
        """
        if not isinstance(hilbert, TensorHilbert):
            raise TypeError(
                "The hilbert space of an EmbedOperator must be a TensorHilbert."
            )
        if not hilbert.subspaces[subspace] == operator.hilbert:
            raise TypeError(
                "The {subspace}-th hilbert space of the tensor hilbert {hilbert} does not match"
                f" the hilbert space of the operator {operator}."
            )

        # operators, coefficients = _flatten_sumoperators(operators, coefficients)

        self._operator = operator
        self._subspace = subspace
        self._dtype = operator.dtype

        super().__init__(
            hilbert,
        )  # forwards all unused arguments so that this class is a mixin.

    @property
    def dtype(self):
        return self._dtype

    @property
    def operator(self) -> AbstractOperator:
        """The tuple of all operators in the terms of this sum. Every
        operator is summed with a corresponding coefficient
        """
        return self._operator

    @property
    def subspace(self) -> int:
        """The index of the subspace in the hilbert space."""
        return self._subspace

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.hilbert} on {self.subspace} : {self.operator})"

    def __add__(self, other):
        if not isinstance(other, EmbedOperator):
            return super().__add__(other)

        if self.hilbert != other.hilbert:
            raise ValueError("Cannot add EmbedOperators with different hilbert spaces.")

        if (
            self.operator.hilbert == other.operator.hilbert
            and self.subspace == other.subspace
        ):
            return EmbedOperator(
                self.hilbert,
                self.operator + other.operator,
                self.subspace,
            )
        else:
            return super().__add__(other)
