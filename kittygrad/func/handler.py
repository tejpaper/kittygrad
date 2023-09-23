from __future__ import annotations

from functools import wraps

import kittygrad.tensor as tsr
import kittygrad.func as func
from ..utils import *


def scalar2tensor(scalar: Scalar,
                  tensor: Tensor,
                  promotion: bool = True,
                  broadcasting: bool = True) -> Tensor:
    dtype = tensor.dtype if promotion else DEFAULT_DTYPE

    if broadcasting:
        return tsr.tensor(np.full(tensor.shape, scalar, dtype=dtype))
    else:
        return tsr.tensor(scalar, dtype=dtype)


def array2tensor(array: np.ndarray,
                 tensor: Tensor,
                 promotion: bool = True,
                 broadcasting: bool = True) -> tuple[Tensor, Tensor]:
    if promotion:
        dtype = np.result_type(array.dtype, tensor.dtype)
        casted = tsr.tensor(array, dtype=dtype)
        tensor = tensor.type(dtype)
    else:
        casted = tsr.tensor(array)

    if broadcasting:
        casted, tensor = func.broadcast_tensors(casted, tensor)

    return casted, tensor


def tensor2tensor(tensor: Tensor,
                  other: Tensor,
                  promotion: bool = True,
                  broadcasting: bool = True) -> tuple[Tensor, Tensor]:
    if promotion:
        dtype = np.result_type(tensor.dtype, other.dtype)
        tensor = tensor.type(dtype)
        other = other.type(dtype)

    if broadcasting:
        tensor, other = func.broadcast_tensors(tensor, other)

    return tensor, other


def autocast(op_symbol: str = None,
             reverse: bool = False,
             promotion: bool = True,
             broadcasting: bool = True,
             prohibited_types: list[type] = None) -> typing.Callable:

    if prohibited_types is None:
        prohibited_types = []

    def handler_decorator(function: typing.Callable) -> typing.Callable:
        @wraps(function)
        def handler(tensor: Tensor, other: Operand, *args, **kwargs) -> Tensor:

            if isinstance(other, Scalar) and Scalar not in prohibited_types:
                other = scalar2tensor(other, tensor, promotion, broadcasting)
            elif isinstance(other, np.ndarray) and np.ndarray not in prohibited_types:
                other, tensor = array2tensor(other, tensor, promotion, broadcasting)
            elif isinstance(other, tsr.Tensor):
                tensor, other = tensor2tensor(tensor, other, promotion, broadcasting)
            else:
                # function
                if op_symbol is None:
                    raise TypeError("Unsupported argument type(s) for {}: 'Tensor' and '{}'."
                                    .format(function.__name__, type(other).__name__))

                # operator
                if reverse:
                    tensor, other = other, tensor

                raise TypeError("Unsupported operand type(s) for {}: '{}' and '{}'."
                                .format(op_symbol, type(tensor).__name__, type(other).__name__))

            return function(tensor, other, *args, **kwargs)
        return handler
    return handler_decorator


def inplace(promotion: bool = True, broadcasting: bool = True, **autocast_kwargs) -> typing.Callable:
    def handler_decorator(function: typing.Callable) -> typing.Callable:

        @wraps(function)
        @autocast(**autocast_kwargs, promotion=False, broadcasting=False)
        def handler(tensor: Tensor, other: Tensor, *args, **kwargs) -> Tensor:

            if tensor._is_leaf and tensor._requires_grad:
                raise RuntimeError("A leaf Variable that requires grad is being used in an in-place operation.")
            elif not tensor._data.flags['WRITEABLE']:
                raise RuntimeError("The inplace operation cannot be applied to a read-only tensor. If this "
                                   "tensor is a view of another, you can try to do the same operation with it.")

            src_tensor = tensor

            if promotion:
                tensor, other = tensor2tensor(tensor, other, promotion=True, broadcasting=False)

                if src_tensor.dtype != tensor.dtype:
                    raise RuntimeError("Output with dtype '{}' doesn't match the promotion dtype '{}'."
                                       .format(src_tensor.dtype, tensor.dtype))

            if broadcasting:
                tensor, other = tensor2tensor(tensor, other, promotion=False, broadcasting=True)

                if src_tensor.shape != tensor.shape:
                    raise RuntimeError("Output with shape {} doesn't match the broadcast shape {}."
                                       .format(src_tensor.shape, tensor.shape))

            tensor._requires_grad |= other.requires_grad
            out = function(tensor, other, *args, **kwargs)

            tensor._version.value += 1
            if out.grad_fn is not None:
                out.grad_fn._versions.out += 1

            return out
        return handler
    return handler_decorator


def view(function: typing.Callable) -> typing.Callable:
    @wraps(function)
    def handler(tensor: Tensor, *args, **kwargs) -> Tensor:
        new_view = function(tensor, *args, **kwargs)
        assert new_view is tensor or new_view._data.base is not None  # TODO: remove me after a bunch of tests
        new_view._version = tensor._version
        return new_view

    return handler
