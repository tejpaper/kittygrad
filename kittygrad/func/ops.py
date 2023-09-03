from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    NegBackward,
    ExpBackward,
    LogBackward,
    AddBackward,
    SubBackward,
    MulBackward,
    DivBackward,
    PowBackward,
)
from ..utils import DotDict

import numpy as np


@backward_graph(NegBackward)
def _neg(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=-tensor._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ExpBackward)
def _exp(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=np.exp(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(LogBackward)
def _log(tensor: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.log(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(AddBackward)
def _add(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=tensor._data + other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(SubBackward)
def _sub(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=tensor._data - other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(MulBackward)
def _mul(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([tensor, other])
    return tsr.tensor(
        data=tensor._data * other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(DivBackward)
def _div(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    other_inv = np.array(1 / other._data)

    ctx.saved_arrays = [other_inv]
    return tsr.tensor(
        data=tensor._data * other_inv,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(PowBackward)
def _pow(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([tensor, other])
    return tsr.tensor(
        data=tensor._data ** other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )
