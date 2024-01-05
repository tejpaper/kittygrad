from __future__ import annotations

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
from kittygrad.utils.classes import DotDict
from kittygrad.utils.functions import *


@BackwardGraph.mount(ToCopyBackward)
def _type(ctx: DotDict, tensor: Tensor, dtype: type | np.dtype) -> Tensor:
    ctx.prev_dtype = tensor.dtype
    return tsr.tensor(
        data=tensor._data,
        dtype=dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(CloneBackward)
def _clone(_ctx: DotDict, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=tensor._data.copy(),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(NegBackward)
def _neg(_ctx: DotDict, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=-tensor._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(AbsBackward)
def _abs(ctx: DotDict, tensor: Tensor) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.abs(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(ExpBackward)
def _exp(_ctx: DotDict, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=np.exp(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(LogBackward)
def _log(ctx: DotDict, tensor: Tensor) -> Tensor:
    ctx.saved_tensors.append(tensor)
    return tsr.tensor(
        data=np.log(tensor._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(AddBackward)
def _add(_ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    return tsr.tensor(
        data=np.add(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(AddBackward)
def _iadd(_ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    np.add(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@BackwardGraph.mount(SubBackward)
def _sub(_ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    return tsr.tensor(
        data=np.subtract(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(SubBackward)
def _isub(_ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad
    np.subtract(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@BackwardGraph.mount(MulBackward)
def _mul(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.multiply(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IMulBackward)
def _imul(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    ctx.saved_arrays = [
        tensor._data.copy() if other.requires_grad else None,
        other._data.copy() if tensor.requires_grad else None,
    ]

    np.multiply(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)
    return tensor


@BackwardGraph.mount(DivBackward)
def _div(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    other_inv = np.divide(1, other._data, dtype=other.dtype)
    ctx.other_inv = other_inv

    return tsr.tensor(
        data=np.multiply(tensor._data, other_inv, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IDivBackward)
def _idiv(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    other_inv = np.divide(1, other._data, dtype=other.dtype)

    if tensor.requires_grad:
        ctx.other_inv = other_inv

    np.multiply(tensor._data, other_inv, out=tensor._data, **NP_OPS_CONFIG)

    if other.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@BackwardGraph.mount(PowBackward)
def _pow(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        tensor,  # always needed (see PowBackward)
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.power(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@BackwardGraph.mount(IPowBackward)
def _ipow(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    tensor._requires_grad |= other.requires_grad

    if tensor.requires_grad:
        ctx.saved_arrays = [tensor._data.copy(), other._data.copy()]

    np.power(tensor._data, other._data, out=tensor._data, **NP_OPS_CONFIG)

    if tensor.requires_grad:
        ctx.out_array = tensor._data.copy()

    return tensor


@BackwardGraph.mount(SumBackward)
def _sum(ctx: DotDict, tensor: Tensor, dim: int | Size | None, keepdim: bool) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=np.sum(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(MeanBackward)
def _mean(ctx: DotDict, tensor: Tensor, dim: int | Size | None, keepdim: bool) -> Tensor:
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    ctx.shape = tensor.shape
    ctx.dim = dim
    return tsr.tensor(
        data=np.mean(tensor._data, axis=dim, keepdims=keepdim),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(VarBackward)
def _var(ctx: DotDict, tensor: Tensor, dim: int | Size | None, correction: int, keepdim: bool) -> Tensor:
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


@BackwardGraph.mount(StdBackward)
def _std(*args, **kwargs) -> Tensor:
    out = _var.__wrapped__(*args, **kwargs)
    np.sqrt(out._data, out=out._data)
    return out


@BackwardGraph.mount(MmBackward)
def _mm(ctx: DotDict, tensor: Tensor, other: Tensor) -> Tensor:
    ctx.saved_tensors.extend([
        tensor if other.requires_grad else None,
        other if tensor.requires_grad else None,
    ])
    return tsr.tensor(
        data=np.matmul(tensor._data, other._data, **NP_OPS_CONFIG),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    )


@autocast(broadcasting=False, prohibited_types=[Scalar])
def mm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    """
    Perform matrix multiplication between two tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.

    mat2 : np.ndarray or Tensor
        The second input tensor or NumPy array to be multiplied with the first tensor.

    Returns
    -------
    Tensor
        A new tensor resulting from the matrix multiplication.
    """
    if input.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError(f"2D tensors expected, but got {input.ndim}D and {mat2.ndim}D tensors.")
    elif input.shape[-1] != mat2.shape[0]:
        raise RuntimeError("input and mat2 shapes cannot be multiplied ({}x{} and {}x{})."
                           .format(*input.shape, *mat2.shape))

    return _mm(input, mat2)


@BackwardGraph.mount(DotBackward)
def _dot(*args, **kwargs) -> Tensor:
    return _mm.__wrapped__(*args, **kwargs)


@autocast(broadcasting=False, prohibited_types=[Scalar])
def dot(input: Tensor, other: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    """
    Compute the dot product between two tensors.

    Parameters
    ----------
    input : Tensor
        The first input tensor.

    other : np.ndarray or Tensor
        The second input tensor or NumPy array for computing the dot product.

    Returns
    -------
    Tensor
        A new tensor resulting from the dot product operation.
    """
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


@autocast(broadcasting=False, prohibited_types=[Scalar])
def mv(input: Tensor, vec: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    """
    Perform matrix-vector multiplication.

    Parameters
    ----------
    input : Tensor
        The input matrix tensor.

    vec : np.ndarray or Tensor
        The input vector tensor or NumPy array to be multiplied with the matrix.

    Returns
    -------
    Tensor
        A new tensor resulting from the matrix-vector multiplication.
    """
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


@autocast(broadcasting=False, prohibited_types=[Scalar])
def bmm(input: Tensor, mat2: np.ndarray | Tensor) -> Tensor:  # noqa: torch-like API
    """
    Batch matrix multiplication.

    Parameters
    ----------
    input : Tensor
        The first input tensor representing a batch of matrices.

    mat2 : np.ndarray or Tensor
        The second input tensor or NumPy array representing a batch of matrices to be multiplied
        with the first batch.

    Returns
    -------
    Tensor
        A new tensor (batch of matrices) resulting from the batch matrix multiplication.
    """
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
    """
    Generalized matrix multiplication for tensors and NumPy arrays.

    Parameters
    ----------
    input : Tensor
        The first input tensor.

    other : np.ndarray or Tensor
        The second input tensor or NumPy array.

    Returns
    -------
    Tensor
        A new tensor resulting from the matrix multiplication.

    Raises
    ------
    RuntimeError
        If input tensors are scalars.

    Notes
    -----
    This function handles matrix multiplication for tensors with different dimensions:
    - If both input tensors are 1-dimensional, it performs a dot product.
    - If both input tensors are 2-dimensional, it performs a matrix multiplication.
    - If one input tensor is 1-dimensional and the other is 2-dimensional, it performs
      a vector-matrix multiplication.
    - If one input tensor is 2-dimensional and the other is 1-dimensional, it performs
      a matrix-vector multiplication.
    - For other cases, it performs batch matrix multiplication.
    """
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
    batch_dims = np.broadcast_shapes(input.shape[:-2], other.shape[:-2])

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
