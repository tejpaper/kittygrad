from __future__ import annotations

import kittygrad.core as core
import kittygrad.tensor.tensor as tsr
from kittygrad.autograd.engine import BackwardGraph
from kittygrad.autograd.ops import (
    ToCopyBackward,
    CloneBackward,
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
from kittygrad.func.handler import autocast
from kittygrad.func.utils import dim2tuple, separate_dims, check_dims


@BackwardGraph.mount(ToCopyBackward)
def _type(ctx: Context, tensor: Tensor, dtype: type | np.dtype) -> Tensor:
    ctx.prev_dtype = tensor.dtype
    return tsr.tensor(
        data=tensor._data,
        dtype=dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(CloneBackward)
def _clone(_ctx: Context, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=tensor._data.copy(),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(NegBackward)
def _neg(_ctx: Context, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=-tensor._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(AbsBackward)
def _abs(ctx: Context, tensor: Tensor) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=core.strict.abs(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(ExpBackward)
def _exp(_ctx: Context, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=core.strict.exp(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(LogBackward)
def _log(ctx: Context, tensor: Tensor) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=core.strict.log(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(AddBackward)
def _add(_ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    return tsr.tensor(
        data=core.strict.add(tensor._data, other._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(AddBackward)
def _iadd(_ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    core.strict.add(tensor._data, other._data, out=tensor._data)
    return tensor


@BackwardGraph.mount(SubBackward)
def _sub(_ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    return tsr.tensor(
        data=core.strict.subtract(tensor._data, other._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(SubBackward)
def _isub(_ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    core.strict.subtract(tensor._data, other._data, out=tensor._data)
    return tensor


@BackwardGraph.mount(MulBackward)
def _mul(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=core.strict.multiply(tensor._data, other._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IMulBackward)
def _imul(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    ctx.saved_arrays = [
        tensor._data.copy() if other.requires_grad else None,
        other._data.copy() if tensor.requires_grad else None,
    ]

    core.strict.multiply(tensor._data, other._data, out=tensor._data)
    return tensor


@BackwardGraph.mount(DivBackward)
def _div(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    other_inv = core.np.divide(1, other._data, dtype=other.dtype)
    ctx.other_inv = other_inv

    return tsr.tensor(
        data=core.strict.multiply(tensor._data, other_inv),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IDivBackward)
def _idiv(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    other_inv = core.np.divide(1, other._data, dtype=other.dtype)

    if tensor.requires_grad:
        ctx.other_inv = other_inv

    core.strict.multiply(tensor._data, other_inv, out=tensor._data)

    if other.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@BackwardGraph.mount(PowBackward)
def _pow(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        tensor,  # always needed (see PowBackward)
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=core.strict.power(tensor._data, other._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IPowBackward)
def _ipow(ctx: Context, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    if tensor.requires_grad:
        ctx.saved_arrays = [tensor._data.copy(), other._data.copy()]

    core.strict.power(tensor._data, other._data, out=tensor._data)

    if tensor.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@BackwardGraph.mount(SumBackward)
def _sum(ctx: Context, tensor: Tensor, dim: int | Size | None, keepdim: bool) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=core.np.sum(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(MeanBackward)
def _mean(ctx: Context, tensor: Tensor, dim: int | Size | None, keepdim: bool) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=core.np.mean(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(VarBackward)
def _var(ctx: Context, tensor: Tensor, dim: int | Size | None, correction: int, keepdim: bool) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    if not tensor.requires_grad:
        return tsr.tensor(
            data=core.np.var(tensor._data, axis=dim, ddof=correction, keepdims=keepdim),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    residuals = tensor._data - core.np.mean(tensor._data, axis=dim, keepdims=True)
    expanded_shape, reps = separate_dims(tensor.shape, dim)
    n = core.np.prod(reps, dtype=tensor.dtype)

    ctx.dim = dim
    ctx.correction = correction
    ctx.residuals = residuals
    ctx.expanded_shape = expanded_shape
    ctx.reps = reps
    ctx.n = n

    return tsr.tensor(
        data=core.np.sum(core.strict.square(residuals), axis=dim, keepdims=keepdim) / (n - correction),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(StdBackward)
def _std(*args, **kwargs) -> Tensor:
    out = _var.__wrapped__(*args, **kwargs)
    core.strict.sqrt(out._data, out=out._data)
    return out


@BackwardGraph.mount(MmBackward)
def _mm(ctx: Context, input: Tensor, mat2: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        input if mat2.requires_grad else None,
        mat2 if input.requires_grad else None,
    ])
    return tsr.tensor(
        data=core.strict.matmul(input._data, mat2._data),
        dtype=input.dtype,
        requires_grad=input.requires_grad or mat2.requires_grad,
    )


@autocast(broadcasting=False, prohibited_types=[core.Scalar])
def mm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:
    if input.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError(f"2D tensors expected, but got {input.ndim}D and {mat2.ndim}D tensors.")
    elif input.shape[-1] != mat2.shape[0]:
        raise RuntimeError("input and mat2 shapes cannot be multiplied ({}x{} and {}x{})."
                           .format(*input.shape, *mat2.shape))

    return _mm(input, mat2)


@BackwardGraph.mount(DotBackward)
def _dot(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[core.Scalar])
def dot(input: Tensor, other: np.ndarray | Tensor) -> Tensor:
    if input.ndim != 1 or other.ndim != 1:
        raise RuntimeError(f"1D tensors expected, but got {input.ndim}D and {other.ndim}D tensors.")
    elif input.nelement() != other.nelement():
        raise RuntimeError("Inconsistent tensor size, expected tensor input and other to have "
                           "the same number of elements, but got {} and {} elements respectively."
                           .format(input.nelement(), other.nelement()))

    return _dot(input, other)


@BackwardGraph.mount(MvBackward)
def _mv(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[core.Scalar])
def mv(input: Tensor, vec: np.ndarray | Tensor) -> Tensor:
    if input.ndim != 2:
        raise RuntimeError(f"input must be a matrix, not a {input.ndim}D tensor.")
    elif vec.ndim != 1:
        raise RuntimeError(f"vec must be a vector, not a {vec.ndim}D tensor.")
    elif input.shape[-1] != vec.nelement():
        raise RuntimeError("input and vec shapes cannot be multiplied ({}x{} and {})."
                           .format(*input.shape, vec.nelement()))

    return _mv(input, vec)


@BackwardGraph.mount(BmmBackward)
def _bmm(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[core.Scalar])
def bmm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:
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


@autocast(broadcasting=False, prohibited_types=[core.Scalar])
def matmul(input: Tensor, other: np.ndarray | Tensor) -> Tensor:
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

    # numpy exceptions are absolutely fine
    batch_dims = core.np.broadcast_shapes(input.shape[:-2], other.shape[:-2])

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
