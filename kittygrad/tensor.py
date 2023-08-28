from __future__ import annotations

import kittygrad.func as func
from .autograd import BackwardAccess
from .utils import *

import numpy as np

import typing
import warnings
from functools import wraps


class Tensor:
    def __init__(self, data, dtype: typing.Optional[type] = None, requires_grad: bool = False) -> None:
        if isinstance(data, np.ndarray):
            if data.dtype not in ALL_DTYPES:
                self._data = data.astype(DEFAULT_DTYPE)
            else:
                self._data = data

        elif dtype is None and ((dtype := np.result_type(*flatten(data))) not in ALL_DTYPES or dtype == np.float_):
            self._data = np.array(data, DEFAULT_DTYPE)
        elif dtype not in ALL_DTYPES:
            raise TypeError(f"Data type '{dtype.__name__}' is not supported")
        else:
            self._data = np.array(data, dtype)

        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None  # points to a node in a backward graph
        self._is_leaf = True
        self._version = 0

    # ============================================= Tensor Representation ==============================================

    def __repr__(self) -> str:
        tensor_prefix = 'tensor('
        tensor_padding = ' ' * len(tensor_prefix)

        array_prefix = 'array('
        array_padding = ' ' * len(array_prefix)

        data_str = repr(self._data)
        if self.ndims:
            data_str = data_str[data_str.find('['):data_str.rfind(']') + 1]
        else:
            data_str = data_str[len(array_prefix):data_str.rfind(',')]
        data_str = data_str.replace('\n' + array_padding, '\n' + tensor_padding)

        main_content = tensor_prefix + data_str

        if self.dtype != DEFAULT_DTYPE:
            main_content += f', dtype={self.dtype}'

        if self._grad_fn is not None:
            return f'{main_content}, grad_fn={self._grad_fn})'  # TODO: test (backward)
        elif self._requires_grad:
            return main_content + ', requires_grad=True)'
        else:
            return main_content + ')'

    def __str__(self) -> str:
        return repr(self)

    # ============================================== Getters and Setters ===============================================

    @property  # not writable
    def data(self) -> Tensor:
        return tensor(self._data, requires_grad=False)

    @property  # not writable
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property  # not writable
    def ndims(self) -> int:
        return len(self._data.shape)

    @property  # not writable
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, new_value: bool) -> None:
        if not self._is_leaf:
            raise RuntimeError(  # TODO: test (backward)                               (^v^)
                "You can only change requires_grad flags of leaf variables." + "" if new_value else
                "If you want to use a computed variable in a subgraph that doesn't "
                "require differentiation use var_no_grad = var.detach().")

        self._requires_grad = new_value

    @property
    def grad(self) -> Tensor:
        return self._grad

    @grad.setter
    def grad(self, new_grad: Tensor | None) -> None:
        if new_grad is None:
            self._grad = new_grad
            return  # it's ok if requires_grad=False

        if type(new_grad) != type(self):
            raise TypeError(f"Assigned grad expected to be a Tensor or None "
                            f"but got grad of '{type(new_grad).__name__}'")
        elif id(new_grad) == id(self):
            raise RuntimeError("Can't assign Variable as its own grad")
        elif new_grad.dtype != self.dtype:
            raise RuntimeError("Assigned grad has data of a different type")
        elif new_grad.shape != self.shape:
            raise RuntimeError("Assigned grad has data of a different size")
        else:
            self._grad = new_grad

        # consistency
        if not self._requires_grad:
            warnings.warn("Trying to assign a gradient to a tensor that doesn't need it. "
                          "The requires_grad attribute is set to True")
            self._requires_grad = True

    @property  # not writable
    def grad_fn(self) -> BackwardAccess:
        return self._grad_fn

    @property  # not writable
    def is_leaf(self) -> bool:
        return self._is_leaf

    def __getitem__(self, *args, **kwargs) -> Tensor:
        if self._requires_grad:
            pass  # TODO: SelectBackward
        else:
            return tensor(data=self._data.__getitem__(*args, **kwargs), requires_grad=False)

    def __setitem__(self, key, value) -> None:
        if type(value) == type(self):
            value = value._data  # noqa: friend

        if self._requires_grad:
            pass  # TODO: CopySlices
        else:
            self._data.__setitem__(key, value)

    # ====================================================== Func ======================================================

    def __array_ufunc__(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError("Unsupported operation with NumPy array. Try swapping operands")  # TODO: mb develop

    @staticmethod
    def __operator_handler(op_symbol: str, reverse: bool = False) -> typing.Callable:
        def handler_decor(operator: typing.Callable) -> typing.Callable:

            @wraps(operator)
            def handler(self, other, *args, **kwargs) -> Tensor:
                if args or kwargs:
                    raise RuntimeError("Incorrect use of operator handler")

                if isinstance(other, Scalar):
                    return operator(self, other)

                if isinstance(other, np.ndarray):
                    other = tensor(other, dtype=self.dtype, requires_grad=False)

                if reverse:
                    first_operand, second_operand = other, self
                else:
                    first_operand, second_operand = self, other

                if type(other) == type(self):
                    if other.dtype != self.dtype:
                        raise TypeError("Operands type mismatch: {} != {}"
                                        .format(first_operand.dtype, second_operand.dtype))
                    elif other.shape != self.shape:  # no broadcasting  TODO: mb develop
                        raise RuntimeError("The size of tensor a {} must match the size of tensor b {}"
                                           .format(first_operand.shape, second_operand.shape))

                    return operator(self, other)

                raise TypeError(
                    "Unsupported operand type(s) for {}: '{}' and '{}'"
                    .format(op_symbol, type(first_operand).__name__, type(second_operand).__name__))

            return handler
        return handler_decor

    @__operator_handler(op_symbol='+')
    def __add__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func.add(self, other)

    @__operator_handler(op_symbol='+', reverse=True)
    def __radd__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func.add(self, other)

    @__operator_handler(op_symbol='*')
    def __mul__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func.mul(self, other)

    @__operator_handler(op_symbol='*', reverse=True)
    def __rmul__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func.mul(self, other)

    # ================================================== Interaction ===================================================

    def detach(self):  # unlike torch makes a full copy of a tensor
        return tensor(np.copy(self._data), requires_grad=False)  # TODO: test

    def backward(self, gradient: typing.Optional[Tensor] = None) -> None:
        if not self._requires_grad:
            raise RuntimeError("Tensor does not require grad and does not have a grad_fn")

        pass  # TODO


tensor = Tensor
