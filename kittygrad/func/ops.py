from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd.engine import backward_graph
from ..autograd.ops import (
    ToCopyBackward,
    NegBackward,
    AbsBackward,
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
    VarBackward,
    StdBackward,
    MmBackward,
    DotBackward,
    MvBackward,
    BmmBackward,
)
from .handler import autocast
from ..utils import *


@backward_graph(ToCopyBackward)
def _type(tensor: Tensor, dtype: type | np.dtype, ctx: DotDict[str, list]) -> Tensor:
    ctx.prev_dtype = tensor.dtype
    return tsr.tensor(
        data=tensor._data,
        dtype=dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(NegBackward)
def _neg(tensor: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    return tsr.tensor(
        data=-tensor._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(AbsBackward)
def _abs(tensor: Tensor, ctx: DotDict[str, list]) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.abs(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ExpBackward)
def _exp(tensor: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    return tsr.tensor(
        data=np.exp(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(LogBackward)
def _log(tensor: Tensor, ctx: DotDict[str, list]) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.log(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(AddBackward)
def _add(tensor: Tensor, other: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    return tsr.tensor(
        data=np.add(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(AddBackward)
def _iadd(tensor: Tensor, other: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    np.add(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@backward_graph(SubBackward)
def _sub(tensor: Tensor, other: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    return tsr.tensor(
        data=np.subtract(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(SubBackward)
def _isub(tensor: Tensor, other: Tensor, _ctx: DotDict[str, list]) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    np.subtract(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@backward_graph(MulBackward)
def _mul(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.multiply(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IMulBackward)
def _imul(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    ctx.saved_arrays = [
        tensor._data.copy() if other.requires_grad else None,
        other._data.copy() if tensor.requires_grad else None,
    ]

    np.multiply(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@backward_graph(DivBackward)
def _div(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    other_inv = np.divide(1, other._data, dtype=other.dtype)
    ctx.other_inv = other_inv

    return tsr.tensor(
        data=np.multiply(tensor._data, other_inv, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IDivBackward)
def _idiv(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    other_inv = np.divide(1, other._data, dtype=other.dtype)

    tensor._requires_grad |= other.requires_grad
    if tensor.requires_grad:
        ctx.other_inv = other_inv

    np.multiply(tensor._data, other_inv, out=tensor._data, **NP_OPS_CONFIG)

    if other.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@backward_graph(PowBackward)
def _pow(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    ctx.saved_tensors.extend([
        tensor,  # always needed (see PowBackward)
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.power(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@backward_graph(IPowBackward)
def _ipow(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    if tensor.requires_grad:
        ctx.saved_arrays = [tensor._data.copy(), other._data.copy()]

    np.power(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)

    if tensor.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@backward_graph(SumBackward)
def _sum(tensor: Tensor, dim: int | Size | None, keepdim: bool, ctx: DotDict[str, list]) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=np.sum(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(MeanBackward)
def _mean(tensor: Tensor, dim: int | Size | None, keepdim: bool, ctx: DotDict[str, list]) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=np.mean(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(VarBackward)
def _var(tensor: Tensor, dim: int | Size | None, correction: int, keepdim: bool, ctx: DotDict[str, list]) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    if not tensor.requires_grad:
        return tsr.tensor(
            data=np.var(tensor._data, axis=dim, ddof=correction, keepdims=keepdim),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    residuals = tensor._data - np.mean(tensor._data, axis=dim, keepdims=True)
    expanded_shape, reps = separate_dims(tensor.shape, dim)
    n = np.prod(reps, dtype=tensor.dtype)

    ctx.dim = dim
    ctx.correction = correction
    ctx.residuals = residuals
    ctx.expanded_shape = expanded_shape
    ctx.reps = reps
    ctx.n = n

    return tsr.tensor(
        data=np.sum(residuals ** 2, axis=dim, keepdims=keepdim) / (n - correction),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@backward_graph(StdBackward)
def _std(*args, **kwargs) -> Tensor:
    out = _var.__wrapped__(*args, **kwargs)
    np.sqrt(out._data, out=out._data)
    return out


@backward_graph(MmBackward)
def _mm(tensor: Tensor, other: Tensor, ctx: DotDict[str, list]) -> Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.matmul(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@autocast(broadcasting=False, prohibited_types=[Scalar])  # TODO: test autocast of these functions
def mm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    if input.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError(f"2D tensors expected, but got {input.ndim}D and {mat2.ndim}D tensors.")
    elif input.shape[-1] != mat2.shape[0]:
        raise RuntimeError("input and mat2 shapes cannot be multiplied ({}x{} and {}x{})."
                           .format(*input.shape, *mat2.shape))

    return _mm(input, mat2)


@backward_graph(DotBackward)
def _dot(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[Scalar])
def dot(input: Tensor, other: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    if input.ndim != 1 or other.ndim != 1:
        raise RuntimeError(f"1D tensors expected, but got {input.ndim}D and {other.ndim}D tensors.")
    elif input.nelement() != other.nelement():
        raise RuntimeError("Inconsistent tensor size, expected tensor input and other to have "
                           "the same number of elements, but got {} and {} elements respectively."
                           .format(input.nelement(), other.nelement()))

    return _dot(input, other)


@backward_graph(MvBackward)
def _mv(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[Scalar])
def mv(input: Tensor, vec: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    if input.ndim != 2:
        raise RuntimeError(f"input must be a matrix, not a {input.ndim}D tensor.")
    elif vec.ndim != 1:
        raise RuntimeError(f"vec must be a vector, not a {vec.ndim}D tensor.")
    elif input.shape[-1] != vec.nelement():
        raise RuntimeError("input and vec shapes cannot be multiplied ({}x{} and {})."
                           .format(*input.shape, vec.nelement()))

    return _mv(input, vec)


@backward_graph(BmmBackward)
def _bmm(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[Scalar])
def bmm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    input_batch_dims = input.shape[:-2]
    mat2_batch_dims = mat2.shape[:-2]

    if input_batch_dims != mat2_batch_dims:
        raise RuntimeError("Batch dimensions of both tensors must be equal, but got "
                           f"{input_batch_dims} and {mat2_batch_dims} respectively.")
    elif not input_batch_dims:
        raise RuntimeError("The batch matrix-matrix product requires the "
                           "tensors to have at least 3 dimensions each.")
    elif input.shape[-1] != mat2.shape[-2]:
        raise RuntimeError("input and mat2 matrix shapes cannot be multiplied ({}x{} and {}x{})."
                           .format(*input.shape[-2:], *mat2.shape[-2:]))

    return _bmm(input, mat2)


@autocast(broadcasting=False, prohibited_types=[Scalar])
def matmul(input: Tensor, other: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError("Input tensors must not be scalars.")

    if input.ndim == 1 and other.ndim == 1:
        return dot.__wrapped__(input, other)
    elif input.ndim == 2 and other.ndim == 2:
        return mm.__wrapped__(input, other)
    elif input.ndim == 1 and other.ndim == 2:
        return mm.__wrapped__(input.unsqueeze(0), other).squeeze(0)
    elif input.ndim == 2 and other.ndim == 1:
        return mv.__wrapped__(input, other)

    batch_dims = np.broadcast_shapes(input.shape[:-2], other.shape[:-2])

    assert not (input.ndim == 1 and other.ndim == 1)  # TODO: remove me after a bunch of tests

    if input.ndim == 1:
        return bmm.__wrapped__(input.unsqueeze(-2).expand(*batch_dims, -1, -1),
                               other.expand(*batch_dims, -1, -1)
                               ).squeeze(-2)
    elif other.ndim == 1:
        return bmm.__wrapped__(input.expand(*batch_dims, -1, -1),
                               other.unsqueeze(-1).expand(*batch_dims, -1, -1)
                               ).squeeze(-1)
    else:
        return bmm.__wrapped__(input.expand(*batch_dims, -1, -1),
                               other.expand(*batch_dims, -1, -1))
