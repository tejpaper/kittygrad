from __future__ import annotations

from functools import wraps

import kittygrad.tensor as tsr
import kittygrad.func as func
from ..utils import *


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
            nonlocal promotion

            if isinstance(other, Scalar) and Scalar not in prohibited_types:
                other = tsr.tensor(np.full(tensor.shape, other, dtype=tensor.dtype))
                return function(tensor, other, *args, **kwargs)

            if isinstance(other, np.ndarray) and np.ndarray not in prohibited_types:
                other = tsr.tensor(other, dtype=tensor.dtype)
                promotion = False

            if isinstance(other, tsr.Tensor):
                if promotion:
                    cast_to = np.result_type(tensor.dtype, other.dtype)
                    tensor = tensor.type(cast_to)
                    other = other.type(cast_to)

                if broadcasting:
                    tensor, other = func.broadcast_tensors(tensor, other)

                return function(tensor, other, *args, **kwargs)

            # function
            if op_symbol is None:
                raise TypeError("Unsupported argument type(s) for {}: 'Tensor' and '{}'."
                                .format(function.__name__, type(other).__name__))

            # operator
            if reverse:
                tensor, other = other, tensor

            raise TypeError("Unsupported operand type(s) for {}: '{}' and '{}'."
                            .format(op_symbol, type(tensor).__name__, type(other).__name__))

        return handler
    return handler_decorator


def inplace(*autocast_args, **autocast_kwargs) -> typing.Callable:
    def handler_decorator(function: typing.Callable) -> typing.Callable:

        @wraps(function)
        @autocast(*autocast_args, **autocast_kwargs, promotion=False, broadcasting=False)
        def handler(tensor: Tensor, other: Tensor, *args, **kwargs) -> Tensor:

            if tensor._is_leaf and tensor._requires_grad:
                raise RuntimeError("A leaf Variable that requires grad is being used in an in-place operation.")
            elif not tensor._data.flags['WRITEABLE']:
                raise RuntimeError("The inplace operation cannot be applied to a read-only tensor. If this "
                                   "tensor is a view of another, you can try to do the same operation with it.")

            cast_to = np.result_type(tensor.dtype, other.dtype)

            if tensor.dtype != cast_to:
                raise RuntimeError("Output with dtype '{}' doesn't match the broadcast dtype '{}'."
                                   .format(tensor.dtype, cast_to))

            prev_shape = tensor.shape
            tensor, other = func.broadcast_tensors(tensor, other.type(cast_to))

            if tensor.shape != prev_shape:
                raise RuntimeError("Output with shape {} doesn't match the broadcast shape {}."
                                   .format(prev_shape, tensor.shape))

            out = function(tensor, other, *args, **kwargs)

            tensor._version.value += 1
            if out.grad_fn is not None:
                out.grad_fn._versions.out += 1

            return out
        return handler
    return handler_decorator


def view(function: typing.Callable) -> typing.Callable:
    @wraps(function)
    def handler(self, *args, **kwargs) -> Tensor:
        new_view = function(self, *args, **kwargs)
        new_view._version = self._version
        return new_view

    return handler
