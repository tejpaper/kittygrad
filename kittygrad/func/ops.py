from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    AddBackward,
    MulBackward,
)


# noinspection PyProtectedMember
@backward_graph(AddBackward)
def add(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=tensor._data + other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ),


# noinspection PyProtectedMember
@backward_graph(MulBackward)
def mul(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=tensor._data * other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ), tensor, other
