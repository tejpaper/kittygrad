from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    TransposeBackward,
    PermuteBackward,
    SqueezeBackward,
    UnsqueezeBackward,
)
from ..utils import *

import numpy as np


@backward_graph(TransposeBackward)
def _transpose(tensor: tsr.Tensor, dim0: int, dim1: int, ctx: DotDict[str, list]) -> tsr.Tensor:
    if tensor.ndim == 0:
        raise RuntimeError("Scalar cannot be transposed.")

    check_dim(dim0, tensor.ndim)
    check_dim(dim1, tensor.ndim)

    ctx.dim0 = dim0
    ctx.dim1 = dim1
    return tsr.tensor(
        data=np.swapaxes(tensor._data, dim0, dim1),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(PermuteBackward)
def _permute(tensor: tsr.Tensor, dims: Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    if tensor.ndim != len(dims):
        raise RuntimeError("Number of dimensions in the tensor input does not match "
                           "the length of the desired ordering of dimensions i.e. "
                           f"input.dim() = {tensor.ndim} is not equal to len(dims) = {len(dims)}.")

    check_dims(dims, tensor.ndim)

    ctx.dims = dims
    return tsr.tensor(
        data=np.transpose(tensor._data, dims),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(SqueezeBackward)
def _squeeze(tensor: tsr.Tensor, dim: int | Size | None, ctx: DotDict[str, list]) -> tsr.Tensor:
    check_dims(dim, tensor.ndim)

    if isinstance(dim, int):
        dim = [dim]

    if dim is not None:
        dim = tuple(d for d in dim if tensor.shape[d] == 1)

    ctx.shape = tensor.shape
    return tsr.tensor(
        data=tensor._data.squeeze(dim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(UnsqueezeBackward)
def _unsqueeze(tensor: tsr.Tensor, dim: int | Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    check_dims(dim, tensor.ndim)

    ctx.dim = dim
    return tsr.tensor(
        data=np.expand_dims(tensor._data, dim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )
