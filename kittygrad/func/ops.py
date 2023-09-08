from __future__ import annotations

from ..autograd import (
    backward_graph,
    NegBackward,
    ExpBackward,
    LogBackward,
    AddBackward,
    SubBackward,
    MulBackward,
    IMulBackward,
    DivBackward,
    IDivBackward,
    PowBackward,
    IPowBackward,
    SumBackward,
    MeanBackward,
    DotBackward,
    MmBackward,
    MvBackward,
)
from ..utils import *

import numpy as np


@backward_graph(NegBackward)
def _neg(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=-tensor._data,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ExpBackward)
def _exp(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=np.exp(tensor._data),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(LogBackward)
def _log(tensor: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.log(tensor._data),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(AddBackward)
def _add(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=np.add(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(AddBackward)
def _iadd(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    tensor._requires_grad |= other.requires_grad
    np.add(tensor._data, other._data, out=tensor._data)
    return tensor


@backward_graph(SubBackward)
def _sub(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=np.subtract(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(SubBackward)
def _isub(tensor: tsr.Tensor, other: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    tensor._requires_grad |= other.requires_grad
    np.subtract(tensor._data, other._data, out=tensor._data)
    return tensor


@backward_graph(MulBackward)
def _mul(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])

    return tsr.tensor(
        data=np.multiply(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IMulBackward)
def _imul(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    tensor._requires_grad |= other.requires_grad

    ctx.saved_arrays = [
        np.copy(tensor._data) if other.requires_grad else None,
        np.copy(other._data) if tensor.requires_grad else None,
    ]

    np.multiply(tensor._data, other._data, out=tensor._data)
    return tensor


@backward_graph(DivBackward)
def _div(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    other_inv = np.array(1 / other._data)
    ctx.other_inv = other_inv

    return tsr.tensor(
        data=np.multiply(tensor._data, other_inv),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IDivBackward)
def _idiv(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    tensor._requires_grad |= other.requires_grad
    other_inv = np.array(1 / other._data)
    np.multiply(tensor._data, other_inv, out=tensor._data)

    ctx.other_inv = other_inv
    ctx.out_array = np.copy(tensor._data)
    return tensor


@backward_graph(PowBackward)
def _pow(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([
        tensor,  # always needed (see PowBackward)
        other if tensor.requires_grad else None,
    ])

    return tsr.tensor(
        data=np.power(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IPowBackward)
def _ipow(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    tensor._requires_grad |= other.requires_grad

    ctx.saved_arrays = [
        np.copy(tensor._data),  # always needed (see IPowBackward)
        np.copy(other._data) if tensor.requires_grad else None,
    ]

    np.power(tensor._data, other._data, out=tensor._data)
    ctx.out_array = np.copy(tensor._data)
    return tensor


@backward_graph(SumBackward)
def _sum(tensor: tsr.Tensor, dim: int | Size | None, keepdim: bool, ctx: DotDict[str, list]) -> tsr.Tensor:
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    ctx.keepdim = keepdim
    return tsr.tensor(
        data=np.sum(tensor._data, axis=dim, keepdims=keepdim),
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
        requires_grad=tensor.requires_grad,
    )


@backward_graph(DotBackward)
def _dot(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.dot(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


def dot(input: tsr.Tensor, other: tsr.Tensor) -> tsr.Tensor:  # noqa: torch-like API
    check_types(input, other)

    if input.ndim != 1 or other.ndim != 1:
        raise RuntimeError(f"1D tensors expected, but got {input.ndim}D and {other.ndim}D tensors.")
    elif input.nelement() != other.nelement():
        raise RuntimeError("Inconsistent tensor size, expected tensor input and other to have "
                           "the same number of elements, but got {} and {} elements respectively."
                           .format(input.nelement(), other.nelement()))

    return _dot(input, other)


@backward_graph(MmBackward)
def _mm(tensor: tsr.Tensor, other: tsr.Tensor, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.matmul(tensor._data, other._data),
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


def mm(input: tsr.Tensor, mat2: tsr.Tensor) -> tsr.Tensor:  # noqa: torch-like API
    check_types(input, mat2)

    if input.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError(f"2D tensors expected, but got {input.ndim}D and {mat2.ndim}D tensors.")
    elif input.shape[-1] != mat2.shape[0]:
        raise RuntimeError("input and mat2 shapes cannot be multiplied ({}x{} and {}x{})."
                           .format(*input.shape, *mat2.shape))

    return _mm(input, mat2)


@backward_graph(MvBackward)
def _mv(*args, **kwargs) -> tsr.Tensor:
    return _mm.__wrapped__(*args, **kwargs)  # DRY


def mv(input: tsr.Tensor, vec: tsr.Tensor) -> tsr.Tensor:  # noqa: torch-like API
    check_types(input, vec)

    if input.ndim != 2:
        raise RuntimeError(f"input must be a matrix, not a {input.ndim}D tensor.")
    elif vec.ndim != 1:
        raise RuntimeError(f"vec must be a vector, not a {vec.ndim}D tensor.")
    elif input.shape[-1] != vec.nelement():
        raise RuntimeError("input and vec shapes cannot be multiplied ({}x{} and {})."
                           .format(*input.shape, vec.nelement()))

    return _mv(input, vec)


def matmul(input: tsr.Tensor, other: tsr.Tensor) -> tsr.Tensor:  # noqa: torch-like API
    if input.ndim == 1 and other.ndim == 1:
        return dot(input, other)
    elif input.ndim == 2 and other.ndim == 2:
        return mm(input, other)
    elif input.ndim == 1 and other.ndim == 2:
        return mm(input.unsqueeze(0), other).squeeze(0)
    elif input.ndim == 2 and other.ndim == 1:
        return mv(input, other)

    pass  # TODO: + exceptions
