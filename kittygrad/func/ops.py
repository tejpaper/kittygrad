from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    AddBackward,
    MulBackward,
)


@backward_graph(AddBackward)
def _add(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=tensor._data + other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ),


@backward_graph(MulBackward)
def _mul(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=tensor._data * other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ), *(
        (other,) if tensor.requires_grad else ()
    ), *(
        (tensor,) if other.requires_grad else ()
    )
