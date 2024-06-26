from __future__ import annotations

import kittygrad.tensor.tensor as tsr
from kittygrad.autograd.engine import BackwardGraph
from kittygrad.autograd.view import (
    TransposeBackward,
    PermuteBackward,
    SqueezeBackward,
    UnsqueezeBackward,
    ExpandBackward,
    IndexBackward,
    IndexPutBackward,
)
from kittygrad.core import *
from kittygrad.func.handler import inplace, share
from kittygrad.func.utils import check_dim, check_dims


@BackwardGraph.mount(TransposeBackward)
def _transpose(ctx: Context, tensor: Tensor, dim0: int, dim1: int) -> Tensor:
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


@BackwardGraph.mount(PermuteBackward)
def _permute(ctx: Context, tensor: Tensor, dims: Size) -> Tensor:
    if tensor.ndim != len(dims):
        raise RuntimeError("Number of dimensions in the tensor input does not match "
                           "the length of the desired ordering of dimensions i.e. "
                           f"input.dim() = {tensor.ndim} is not equal to len(dims) = {len(dims)}.")

    check_dims(dims, tensor.ndim)

    ctx.dims = dims
    return tsr.tensor(
        data=np.transpose(tensor._data, dims),
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(SqueezeBackward)
def _squeeze(ctx: Context, tensor: Tensor, dim: int | Size | None) -> Tensor:
    if isinstance(dim, int):
        check_dim(dim, tensor.ndim)
        dim = (dim,) if tensor.shape[dim] == 1 else ()

    elif dim is not None:
        check_dims(dim, tensor.ndim)
        dim = tuple(d for d in dim if tensor.shape[d] == 1)

    ctx.shape = tensor.shape
    return tsr.tensor(
        data=tensor._data.squeeze(dim),
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(UnsqueezeBackward)
def _unsqueeze(ctx: Context, tensor: Tensor, dim: int | Size) -> Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)

    check_dims(dim, tensor.ndim + len(dim))

    ctx.dim = dim
    return tsr.tensor(
        data=np.expand_dims(tensor._data, dim),
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(ExpandBackward)
def _expand(ctx: Context, tensor: Tensor, shape: Size, expanded_dims: Size, offset: int) -> Tensor:
    ctx.expanded_dims = tuple(expanded_dims)
    ctx.leading_dims = tuple(range(offset))

    return tsr.tensor(
        data=np.broadcast_to(tensor._data, shape),
        requires_grad=tensor.requires_grad,
    )


@share
def broadcast_to(input: Tensor, shape: Size) -> Tensor:
    for dim in shape:
        if dim <= 0 and dim != -1:
            raise RuntimeError(f"The expanded size of the tensor ({dim}) isn't allowed.")

    old_shape = list(input.shape)
    new_shape = list(shape)
    expanded_dims = []

    offset = len(new_shape) - len(old_shape)

    for i in range(offset):
        if new_shape[i] == -1:
            raise RuntimeError("The expanded size of the tensor (-1) isn't allowed "
                               f"in a leading, non-existing dimension {i}.")

        expanded_dims.append(i)

    for i in range(offset, len(new_shape)):
        if new_shape[i] == -1:
            new_shape[i] = old_shape[i - offset]
        elif new_shape[i] != old_shape[i - offset]:
            if old_shape[i - offset] != 1:
                raise RuntimeError("The expanded size of the tensor ({}) must match the existing size ({}) "
                                   "at non-singleton dimension {}. Target sizes: {}. Tensor sizes: {}."
                                   .format(new_shape[i], old_shape[i - offset], i, tuple(shape), tuple(old_shape)))

            expanded_dims.append(i)

    if new_shape == old_shape:
        return input
    else:
        return _expand(input, new_shape, expanded_dims, offset)


def broadcast_tensors(*tensors: Tensor) -> list[Tensor]:
    # numpy exceptions are absolutely fine
    result_shape = np.broadcast(*[t._data for t in tensors]).shape
    return [broadcast_to(t, result_shape) for t in tensors]


@BackwardGraph.mount(IndexBackward)
def _index(ctx: Context, tensor: Tensor, key) -> Tensor:
    if not isinstance(key, tuple):
        key = (key,)
    if not any(ind is Ellipsis for ind in key):
        key = (*key, ...)

    ctx.key = key
    ctx.shape = tensor.shape
    return tsr.tensor(
        data=tensor._data[*key],
        requires_grad=tensor.requires_grad,
    )


@share
def _index_view(*args, **kwargs) -> Tensor:
    return _index(*args, **kwargs)


@inplace(promotion=False, broadcasting=False)
@BackwardGraph.mount(IndexPutBackward)
def _index_put(ctx: Context, tensor: Tensor, value: Operand, key) -> Tensor:
    tensor._data[key] = value._data
    ctx.key = key
    return tensor
