from __future__ import annotations

from .utils import *
import kittygrad.func as func

import numpy as np

import typing
from functools import wraps


class Tensor:
    def __init__(self, data, dtype: typing.Optional[type] = None, requires_grad: bool = False) -> None:
        if isinstance(data, np.ndarray):
            if data.dtype not in ALL_DTYPES:
                self._data = data.astype(DEFAULT_DTYPE)
            else:
                self._data = data

        elif dtype is None and (dtype := np.result_type(*flatten(data))) not in ALL_DTYPES:
            self._data = np.array(data, DEFAULT_DTYPE)

        else:
            self._data = np.array(data, dtype)

        # TODO: getters/setters
        self.requires_grad = requires_grad  # when manually it turns on then is_leaf = True
        self.grad = None
        self.grad_fn = None  # points to a node in a backward graph
        self.is_leaf = True
        self._version = 0

    # ============================================= Tensor Representation ==============================================

    def __str__(self) -> str:
        tensor_prefix = 'tensor('
        tensor_padding = ' ' * len(tensor_prefix)

        array_prefix = 'array('
        array_padding = ' ' * len(array_prefix)

        data_str = repr(self._data)
        data_str = data_str[data_str.find('['):data_str.rfind(']') + 1]
        data_str = data_str.replace('\n' + array_padding, '\n' + tensor_padding)

        return tensor_prefix + data_str + ')'

    def __repr__(self) -> str:
        prefix = self.__str__()[:-1]

        if self.grad_fn is not None:
            return f'{prefix}, grad_fn={self.grad_fn})'  # TODO: test
        elif self.requires_grad:
            return prefix + ', requires_grad=True)'
        else:
            return prefix + ')'

    # ============================================== Getters and Setters ===============================================

    @property
    def data(self) -> Tensor:
        return tensor(self._data, requires_grad=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, *args, **kwargs) -> Tensor:
        if self.requires_grad:
            pass  # TODO: SelectBackward
        else:
            return tensor(data=self._data.__getitem__(*args, **kwargs), requires_grad=False)

    def __setitem__(self, key, value) -> None:
        if type(value) == type(self):
            value = value._data  # noqa: friend

        if self.requires_grad:
            pass  # TODO: CopySlices
        else:
            self._data.__setitem__(key, value)

    # ====================================================== Func ======================================================

    def __array_ufunc__(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError('Unsupported operation with NumPy array. Try swapping operands')  # TODO: mb develop

    @staticmethod
    def __operator_handler(op_symbol: str, reverse: bool = False):
        def handler_decor(operator: typing.Callable):

            @wraps(operator)
            def handler(self, other, *args, **kwargs):
                if args or kwargs:
                    raise RuntimeError('Incorrect use of operator handler')

                if isinstance(other, np.ndarray):
                    other = tensor(other, requires_grad=False)  # TODO: add dtype and its check

                if isinstance(other, Scalar) or type(other) == type(self):
                    return operator(self, other)

                if reverse:
                    first_operand = type(other).__name__
                    second_operand = type(self).__name__
                else:
                    first_operand = type(self).__name__
                    second_operand = type(other).__name__

                raise TypeError(
                    "Unsupported operand type(s) for {}: '{}' and '{}'"
                    .format(op_symbol, first_operand, second_operand))

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

    def backward(self, gradient: typing.Optional[Tensor] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError('Tensor does not require grad and does not have a grad_fn')

        pass  # TODO


tensor = Tensor
