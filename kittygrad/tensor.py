from __future__ import annotations

import kittygrad.func as func
from .autograd import FnBackward, check_locks
from .constants import *
from .utils import flatten

import numpy as np

import typing
import warnings
from functools import wraps

warnings.simplefilter('always', UserWarning)


class Tensor:
    def __init__(self, data, dtype: type | np.dtype | None = None, requires_grad: bool = False) -> None:
        if isinstance(data, np.ndarray):
            if data.dtype not in ALL_DTYPES:
                self._data = data.astype(DEFAULT_DTYPE)
            else:
                self._data = data

        elif dtype is None and ((dtype := np.result_type(*flatten(data))) not in ALL_DTYPES or dtype == np.float_):
            self._data = np.array(data, DEFAULT_DTYPE)
        elif dtype not in ALL_DTYPES:
            raise TypeError(f"Data type '{dtype.__name__}' is not supported.")
        else:
            self._data = np.array(data, dtype)

        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None  # points to a node in a backward graph
        self._is_leaf = True
        self._retains_grad = False
        self._version = 0

    # ============================================= Tensor Representation ==============================================

    def __repr__(self) -> str:
        tensor_prefix = 'tensor('
        tensor_padding = ' ' * len(tensor_prefix)

        array_prefix = 'array('
        array_padding = ' ' * len(array_prefix)

        data_str = repr(self._data)
        if self.ndim:
            data_str = data_str[data_str.find('['):data_str.rfind(']') + 1]
        else:
            data_str = data_str[len(array_prefix):data_str.rfind(',')]
        data_str = data_str.replace('\n' + array_padding, '\n' + tensor_padding)

        main_content = tensor_prefix + data_str

        if self.dtype != DEFAULT_DTYPE:
            main_content += f', dtype={self.dtype}'

        if self._grad_fn is not None:
            return f'{main_content}, grad_fn={self._grad_fn})'
        elif self._requires_grad:
            return main_content + ', requires_grad=True)'
        else:
            return main_content + ')'

    def __str__(self) -> str:
        return repr(self)

    # ============================================== Getters and Setters ===============================================

    @property  # not writable
    def data(self) -> Tensor:
        return tensor(self._data)

    @property  # not writable
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property  # not writable
    def ndim(self) -> int:
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
    def grad(self) -> Tensor | None:
        if not self._is_leaf and not self._retains_grad:  # TODO: test
            warnings.warn("The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. "
                          "Its .grad attribute won't be populated during autograd.backward(). If you "
                          "indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() "
                          "on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you "
                          "access the leaf Tensor instead.")

        if self._grad is not None:
            return tensor(self._grad)

    @grad.setter
    def grad(self, new_grad: Tensor | None) -> None:
        if new_grad is None:
            self._grad = new_grad
            return  # it's ok if requires_grad=False

        if type(new_grad) != type(self):
            raise TypeError(f"Assigned grad expected to be a Tensor or None "
                            f"but got grad of '{type(new_grad).__name__}'.")
        elif id(new_grad) == id(self):
            raise RuntimeError("Can't assign Variable as its own grad.")
        elif new_grad.dtype != self.dtype:
            raise TypeError("Assigned grad has data of a different type.")
        elif new_grad.shape != self.shape:
            raise RuntimeError("Assigned grad has data of a different size.")
        else:
            self._grad = new_grad._data

        # consistency
        if not self._requires_grad:
            warnings.warn("Trying to assign a gradient to a tensor that doesn't need it. "
                          "The requires_grad attribute is set to True.")
            self._requires_grad = True

    @property  # not writable
    def grad_fn(self) -> FnBackward:
        return self._grad_fn

    @property  # not writable
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property  # one-way writable with tensor.retain_grad()
    def retains_grad(self) -> bool:
        return self._retains_grad

    # ================================================= Generalization =================================================

    def __array_ufunc__(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError("Unsupported operation with NumPy array. Try swapping operands.")  # TODO: mb develop

    @staticmethod
    def __operator_handler(op_symbol: str, reverse: bool = False) -> typing.Callable:
        def handler_decor(operator: typing.Callable) -> typing.Callable:

            @wraps(operator)
            def handler(self, other, *args, **kwargs) -> Tensor:
                if args or kwargs:
                    raise RuntimeError("Incorrect use of operator handler.")

                if isinstance(other, Scalar):
                    other = tensor(np.full(self.shape, other, dtype=self.dtype))
                elif isinstance(other, np.ndarray):
                    other = tensor(other, dtype=self.dtype)

                if reverse:
                    first_operand, second_operand = other, self
                else:
                    first_operand, second_operand = self, other

                if type(other) == type(self):
                    if other.dtype != self.dtype:
                        raise TypeError("Operands type mismatch: {} != {}."
                                        .format(first_operand.dtype, second_operand.dtype))
                    elif other.shape != self.shape:  # TODO: broadcasting
                        raise RuntimeError("The size of tensor a {} must match the size of tensor b {}."
                                           .format(first_operand.shape, second_operand.shape))

                    return operator(self, other)

                raise TypeError("Unsupported operand type(s) for {}: '{}' and '{}'."
                                .format(op_symbol, type(first_operand).__name__, type(second_operand).__name__))

            return handler
        return handler_decor

    @staticmethod
    def __inplace_operation(method: typing.Callable) -> typing.Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._is_leaf and self._requires_grad:
                raise RuntimeError("A leaf Variable that requires grad is being used in an in-place operation.")

            self._version += 1
            return method(self, *args, **kwargs)

        return wrapper

    # ====================================================== Func ======================================================

    def __pos__(self) -> Tensor:
        return self

    def __neg__(self) -> Tensor:
        return func._neg(self)[0]

    def exp(self) -> Tensor:
        return func._exp(self)[0]

    def sigmoid(self) -> Tensor:
        return func._sigmoid(self)[0]

    def tanh(self) -> Tensor:
        return func._tanh(self)[0]

    def relu(self) -> Tensor:
        return func._relu(self)[0]

    @__operator_handler(op_symbol='+')
    def __add__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._add(self, other)[0]

    @__operator_handler(op_symbol='+', reverse=True)
    def __radd__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._add(self, other)[0]

    @__operator_handler(op_symbol='-')
    def __sub__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._sub(self, other)[0]

    @__operator_handler(op_symbol='-', reverse=True)
    def __rsub__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._sub(other, self)[0]

    @__operator_handler(op_symbol='*')
    def __mul__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._mul(self, other)[0]

    @__operator_handler(op_symbol='*', reverse=True)
    def __rmul__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._mul(self, other)[0]

    @__operator_handler(op_symbol='/')
    def __truediv__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._div(self, other)[0]

    @__operator_handler(op_symbol='/', reverse=True)
    def __rtruediv__(self, other: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._div(other, self)[0]

    @__operator_handler(op_symbol='**')
    def __pow__(self, power: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._pow(self, power)[0]

    @__operator_handler(op_symbol='**', reverse=True)
    def __rpow__(self, power: Scalar | np.ndarray | Tensor) -> Tensor:
        return func._pow(power, self)[0]

    # ====================================================== View ======================================================

    def __getitem__(self, *args, **kwargs) -> Tensor:
        if self._requires_grad:
            pass  # TODO: SelectBackward
        else:
            return tensor(data=self._data.__getitem__(*args, **kwargs))

    @__inplace_operation
    def __setitem__(self, key, value) -> None:
        if type(value) == type(self):
            value = value._data

        if self._requires_grad:
            pass  # TODO: CopySlices
        else:
            self._data.__setitem__(key, value)

    # ================================================== Interaction ===================================================

    def detach(self):  # unlike torch makes a full copy of a tensor
        return tensor(np.copy(self._data))

    def retain_grad(self) -> None:
        if not self._is_leaf:
            self._retains_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        if not self._requires_grad:
            raise RuntimeError("Tensor does not require grad and does not have a grad_fn.")
        elif gradient is None and self.ndim:
            raise RuntimeError("Grad can be implicitly created only for scalar outputs.")

        if gradient is None:
            gradient = tensor(1, dtype=self.dtype)

        if self._is_leaf:
            self.grad = gradient
            return

        if self.shape != gradient.shape:
            raise RuntimeError("Assigned grad has data of a different size.")

        if self._grad_fn._lock != 0:
            warnings.warn("A .backward() call from the middle of the computational graph was noticed.")

        self._grad_fn._lock = 1
        temp = self._grad_fn

        self._grad_fn.propagate(gradient._data)

        if check_locks(temp):
            warnings.warn("Backpropagation not completed. The computational graph "
                          "has at least one more output for the .backward() call.")


tensor = Tensor
