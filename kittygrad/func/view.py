from __future__ import annotations

from ..autograd.view import (
    TransposeBackward,
    PermuteBackward,
    SqueezeBackward,
    UnsqueezeBackward,
    ExpandBackward,
)
from ..autograd.engine import backward_graph
from ..utils import *


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
        requires_grad=tensor.requires_grad,
    )


@backward_graph(PermuteBackward)
def _permute(tensor: tsr.Tensor, dims: Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    if tensor.ndim != len(dims):
        raise RuntimeError("Number of dimensions in the tensor input does not match "
                           "the length of the desired ordering of dimensions i.e. "
                           f"input.dim() = {tensor.ndim} is not equal to len(dims) = {len(dims)}.")
    else:
        check_dims(dims, tensor.ndim)

    ctx.dims = dims
    return tsr.tensor(
        data=np.transpose(tensor._data, dims),
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
        requires_grad=tensor.requires_grad,
    )


@backward_graph(UnsqueezeBackward)
def _unsqueeze(tensor: tsr.Tensor, dim: int | Size, ctx: DotDict[str, list]) -> tsr.Tensor:
    if isinstance(dim, int):
        dim = [dim]

    check_dims(dim, tensor.ndim + len(dim))

    ctx.dim = tuple(dim)
    return tsr.tensor(
        data=np.expand_dims(tensor._data, dim),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ExpandBackward)
def _expand(tensor: tsr.Tensor, shape: Size, expanded_dims: Size, offset: int, ctx: DotDict[str, list]) -> tsr.Tensor:
    ctx.expanded_dims = tuple(expanded_dims)
    ctx.leading_dims = tuple(range(offset))

    return tsr.tensor(
        data=np.broadcast_to(tensor._data, shape),
        requires_grad=tensor.requires_grad,
    )


def broadcast_to(input: tsr.Tensor, shape: Size) -> tsr.Tensor:  # noqa: torch-like API
    if any(dim <= 0 and dim != -1 for dim in shape):
        raise RuntimeError(f"The expanded size of the tensor ({min(shape)}) isn't allowed.")  # TODO: bug

    old_shape = list(input.shape)
    new_shape = list(shape)
    expanded_dims = []

    offset = len(new_shape) - len(old_shape)

    for i in range(offset):
        if new_shape[i] == -1:
            raise RuntimeError("The expanded size of the tensor (-1) isn't allowed "
                               f"in a leading, non-existing dimension {i}.")

        expanded_dims.append(i)

    for i in range(len(new_shape) - len(old_shape), len(new_shape)):
        if new_shape[i] == -1:
            new_shape[i] = old_shape[i - offset]
        elif new_shape[i] != old_shape[i - offset]:
            if old_shape[i - offset] != 1:
                raise RuntimeError("The expanded size of the tensor ({}) must match the existing size ({}) "
                                   "at non-singleton dimension {}. Target sizes: {}. Tensor sizes: {}."
                                   .format(new_shape[i], old_shape[i - offset], i, list(shape), old_shape))

            expanded_dims.append(i)

    if new_shape == old_shape:
        return input
    else:
        return _expand(input, new_shape, expanded_dims, offset)


def broadcast_tensors(*tensors: tsr.Tensor) -> list[tsr.Tensor]:
    # numpy exceptions are absolutely fine
    result_shape = np.broadcast(*[t._data for t in tensors]).shape
    return [broadcast_to(t, result_shape) for t in tensors]
