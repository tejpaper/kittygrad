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
    SumBackward,
    MeanBackward,
    DotBackward,
)
from ..utils import *

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


@backward_graph(SumBackward)
def _sum(tensor: tsr.Tensor, dim: int | Size | None, keepdim: bool, ctx: DotDict[str, list]) -> tsr.Tensor:
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    ctx.keepdim = keepdim
    return tsr.tensor(
        data=np.sum(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(MeanBackward)
def _mean(tensor: tsr.Tensor, dim: int | Size | None, keepdim: bool, ctx: DotDict[str, list]) -> tsr.Tensor:
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    ctx.keepdim = keepdim
    return tsr.tensor(
        data=np.mean(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(DotBackward)
def _dot(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([tensor, other])
    return tsr.tensor(
        data=np.dot(tensor._data, other._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


def dot(input: tsr.Tensor, other: tsr.Tensor) -> tsr.Tensor:  # noqa: torch-like API
    if input.ndim != 1 or other.ndim != 1:
        raise RuntimeError(f"1D tensors expected, but got {input.ndim}D and {other.ndim}D tensors.")
    elif input.dtype != other.dtype:
        raise TypeError("Operands type mismatch: {} != {}."
                        .format(input.dtype, other.dtype))
    elif input.nelement() != other.nelement():
        raise RuntimeError("Inconsistent tensor size, expected tensor input and other to have "
                           "the same number of elements, but got {} and {} elements respectively."
                           .format(input.nelement(), other.nelement()))

    return _dot(input, other)
