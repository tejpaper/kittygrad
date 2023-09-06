from __future__ import annotations

from ..autograd import (
    backward_graph,
    TransposeBackward,
    PermuteBackward,
    SqueezeBackward,
    UnsqueezeBackward,
    ExpandBackward,
)
from ..utils import *

import numpy as np


@backward_graph(TransposeBackward)
def _transpose(tensor: tsr.Tensor, dim0: int, dim1: int, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.dim0 = dim0
    ctx.dim1 = dim1
    return tsr.tensor(
        data=np.swapaxes(tensor._data, dim0, dim1),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(PermuteBackward)
def _permute(tensor: tsr.Tensor, dims: Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.dims = dims
    return tsr.tensor(
        data=np.transpose(tensor._data, dims),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(SqueezeBackward)
def _squeeze(tensor: tsr.Tensor, dim: int | Size | None, ctx: DotDict[str, list]) -> tsr.Tensor:
    if isinstance(dim, int):
        dim = [dim]

    if dim is not None:
        dim = tuple(d for d in dim if tensor.shape[d] == 1)

    ctx.shape = tensor.shape
    return tsr.tensor(
        data=tensor._data.squeeze(dim),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(UnsqueezeBackward)
def _unsqueeze(tensor: tsr.Tensor, dim: int | Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.dim = dim
    return tsr.tensor(
        data=np.expand_dims(tensor._data, dim),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ExpandBackward)
def _expand(tensor: tsr.Tensor, sizes: int | Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    pass


def _broadcast_tensors(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor, tsr.Tensor]:
    pass


def broadcast_tensors(*tensors) -> list[tsr.Tensor]:
    pass

