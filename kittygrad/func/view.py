from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    TransposeBackward,
    PermuteBackward,
)
from ..utils import *

import numpy as np


@backward_graph(TransposeBackward)
def _transpose(tensor: tsr.Tensor, dim0: int, dim1: int, ctx: DotDict[str, list]) -> tsr.Tensor:
    if tensor.ndim == 0:
        raise RuntimeError("Scalar cannot be transposed.")

    check_dim(tensor.ndim, dim0)
    check_dim(tensor.ndim, dim1)

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
    elif len(set(dims)) != len(dims):
        raise RuntimeError("Duplicate dims are not allowed.")

    for dim in dims:
        check_dim(tensor.ndim, dim)

    ctx.dims = dims
    return tsr.tensor(
        data=np.transpose(tensor._data, dims),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )
