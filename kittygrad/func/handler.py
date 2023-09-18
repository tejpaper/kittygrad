from __future__ import annotations

from functools import wraps

import kittygrad.func as func
from ..utils import *


def autocast(op_symbol: str = None, reverse: bool = False, dtype: bool = True, shape: bool = True) -> typing.Callable:
    def handler_decorator(function: typing.Callable) -> typing.Callable:
        @wraps(function)
        def handler(tensor: tsr.Tensor, other: tsr.Operand, *args, **kwargs) -> tsr.Tensor:
            nonlocal dtype

            if isinstance(other, Scalar):
                other = tsr.tensor(np.full(tensor.shape, other, dtype=tensor.dtype))
                return function(tensor, other, *args, **kwargs)

            if isinstance(other, np.ndarray):
                other = tsr.tensor(other, dtype=tensor.dtype)
                dtype = False

            if isinstance(other, tsr.Tensor):
                if dtype:
                    cast_to = np.result_type(tensor.dtype, other.dtype)
                    tensor = tensor.type(cast_to)
                    other = other.type(cast_to)

                if shape:
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


def inplace(function: typing.Callable) -> typing.Callable:
    @wraps(function)
    def handler(tensor: tsr.Tensor, other: tsr.Tensor, *args, **kwargs) -> tsr.Tensor:
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


def view(function: typing.Callable) -> typing.Callable:
    @wraps(function)
    def handler(self, *args, **kwargs) -> tsr.Tensor:
        new_view = function(self, *args, **kwargs)
        new_view._version = self._version
        return new_view

    return handler
