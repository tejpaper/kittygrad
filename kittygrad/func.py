from __future__ import annotations

import kittygrad.tensor as tsr
from .utils import *
from .autograd import (
    AccumulateGrad,
    AddBackward,
    MulBackward,
)


# noinspection PyProtectedMember
def add(tensor: tsr.Tensor, other: tsr.Tensor) -> tsr.Tensor:
    out = tsr.tensor(
        data=tensor._data + other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )

    # TODO: DRY
    if not out.requires_grad:
        return out

    if (tensor_grad_fn := tensor.grad_fn) is not None:
        tensor.grad_fn._lock += 1
    elif tensor.is_leaf:
        tensor_grad_fn = AccumulateGrad(tensor)

    if (other_grad_fn := other.grad_fn) is not None:
        other.grad_fn._lock += 1
    elif other.is_leaf:
        other_grad_fn = AccumulateGrad(other)

    out._is_leaf = False
    out._grad_fn = AddBackward(
        source_ctx=[],
        outcome_ctx=out,
        next_functions=[tensor_grad_fn, other_grad_fn],
    )

    return out


# noinspection PyProtectedMember
def mul(tensor: tsr.Tensor, other: tsr.Tensor) -> tsr.Tensor:
    out = tsr.tensor(
        data=tensor._data * other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )

    # TODO: DRY
    if not out.requires_grad:
        return out

    if (tensor_grad_fn := tensor.grad_fn) is not None:
        tensor.grad_fn._lock += 1
    elif tensor.is_leaf:
        tensor_grad_fn = AccumulateGrad(tensor)

    if (other_grad_fn := other.grad_fn) is not None:
        other.grad_fn._lock += 1
    elif other.is_leaf:
        other_grad_fn = AccumulateGrad(other)

    out._is_leaf = False
    out._grad_fn = MulBackward(
        source_ctx=[tensor, other],
        outcome_ctx=out,
        next_functions=[tensor_grad_fn, other_grad_fn],
    )

    return out
