from __future__ import annotations

import typing

import kittygrad.func as func
from .autograd import FnBackward, check_locks
from .constants import *
from .utils import (
    flatten,
    check_types,
    check_dim,
    check_dims,
)

import numpy as np

import warnings
from functools import wraps
from ctypes import c_int64 as mutable_int

np.set_printoptions(precision=4)
warnings.simplefilter('always', UserWarning)


class Tensor:
    def __init__(self, data, dtype: type | np.dtype | None = None, requires_grad: bool = False) -> None:
        if type(data) == type(self):
            raise RuntimeError("If you want to create a new tensor from another, use "
                               "sourceTensor.detach() and then specify the requires_grad attribute.")

        is_ndarray = isinstance(data, np.ndarray)
        dtype_unknown = dtype is None

        if dtype_unknown:
            if not is_ndarray:
                dtype = np.result_type(*flatten(data))
                if dtype == np.float_:
                    dtype = DEFAULT_DTYPE
            else:
                dtype = data.dtype

        supported_dtype = dtype in ALL_DTYPES

        # TODO: test
        if not dtype_unknown and not supported_dtype:
            raise TypeError(f"Data type '{dtype.__name__}' is not supported.")
        elif dtype_unknown and not supported_dtype:
            if is_ndarray:
                warnings.warn(f"Passed NumPy array has an unsupported data type. "
                              f"Created a copy based on '{DEFAULT_DTYPE.__name__}' dtype.")
            self._data = np.array(data, DEFAULT_DTYPE)
        elif supported_dtype and (not is_ndarray or not dtype_unknown):
            self._data = np.array(data, dtype)
        else:
            self._data = data

        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None  # points to a node in a backward graph
        self._is_leaf = True
        self._retains_grad = False
        self._version = mutable_int(0)

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
    def shape(self) -> Size:
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

    @property  # not writable
    def version(self) -> int:
        return self._version.value

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
                    check_types(first_operand, second_operand)

                    if other.shape != self.shape:  # TODO: broadcasting
                        raise RuntimeError("The size of tensor a {} must match the size of tensor b {}."
                                           .format(first_operand.shape, second_operand.shape))

                    return operator(self, other, *args, **kwargs)

                raise TypeError("Unsupported operand type(s) for {}: '{}' and '{}'."
                                .format(op_symbol, type(first_operand).__name__, type(second_operand).__name__))

            return handler
        return handler_decor

    @staticmethod
    def __inplace_operation(method: typing.Callable) -> typing.Callable:  # TODO: test
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._is_leaf and self._requires_grad:
                raise RuntimeError("A leaf Variable that requires grad is being used in an in-place operation.")

            out = method(self, *args, **kwargs)
            self._version.value += 1
            return out

        return wrapper

    @staticmethod
    def __view(method: typing.Callable) -> typing.Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            new_view = method(self, *args, **kwargs)
            new_view._version = self._version
            return new_view

        return wrapper

    # ====================================================== Func ======================================================

    def __pos__(self) -> Tensor:
        return self

    def __neg__(self) -> Tensor:
        return func._neg(self)

    def exp(self) -> Tensor:
        return func._exp(self)

    def log(self) -> Tensor:
        return func._log(self)

    def sigmoid(self) -> Tensor:
        return func._sigmoid(self)

    def tanh(self) -> Tensor:
        return func._tanh(self)

    def relu(self) -> Tensor:
        return func._relu(self)

    @__operator_handler(op_symbol='+')
    def __add__(self, other: Operand) -> Tensor:
        return func._add(self, other)

    @__operator_handler(op_symbol='+', reverse=True)
    def __radd__(self, other: Operand) -> Tensor:
        return func._add(self, other)

    @__operator_handler(op_symbol='-')
    def __sub__(self, other: Operand) -> Tensor:
        return func._sub(self, other)

    @__operator_handler(op_symbol='-', reverse=True)
    def __rsub__(self, other: Operand) -> Tensor:
        return func._sub(other, self)

    @__operator_handler(op_symbol='*')
    def __mul__(self, other: Operand) -> Tensor:
        return func._mul(self, other)

    @__operator_handler(op_symbol='*', reverse=True)
    def __rmul__(self, other: Operand) -> Tensor:
        return func._mul(self, other)

    @__operator_handler(op_symbol='/')
    def __truediv__(self, other: Operand) -> Tensor:
        return func._div(self, other)

    @__operator_handler(op_symbol='/', reverse=True)
    def __rtruediv__(self, other: Operand) -> Tensor:
        return func._div(other, self)

    @__operator_handler(op_symbol='**')
    def __pow__(self, power: Operand) -> Tensor:
        return func._pow(self, power)

    @__operator_handler(op_symbol='**', reverse=True)
    def __rpow__(self, power: Operand) -> Tensor:
        return func._pow(power, self)

    def sum(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return func._sum(self, dim, keepdim)

    def mean(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return func._mean(self, dim, keepdim)

    # ================================================== Inplace Func ==================================================

    @__inplace_operation
    @__operator_handler(op_symbol='+=')
    def __iadd__(self, other: Operand) -> Tensor:
        return func._add(self, other, inplace=True)

    @__inplace_operation
    @__operator_handler(op_symbol='-=')
    def __isub__(self, other: Operand) -> Tensor:
        return func._sub(self, other, inplace=True)

    @__inplace_operation
    @__operator_handler(op_symbol='*=')
    def __imul__(self, other: Operand) -> Tensor:
        return func._mul(self, other, inplace=True)

    @__inplace_operation
    @__operator_handler(op_symbol='/=')
    def __itruediv__(self, other: Operand) -> Tensor:
        return func._div(self, other, inplace=True)

    @__inplace_operation
    @__operator_handler(op_symbol='**=')
    def __ipow__(self, other: Operand) -> Tensor:
        return func._pow(self, other, inplace=True)

    # ====================================================== View ======================================================

    @__view
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        if self.ndim == 0:
            raise RuntimeError("Scalar cannot be transposed.")

        check_dim(dim0, self.ndim)
        check_dim(dim1, self.ndim)

        return func._transpose(self, dim0, dim1)

    @property
    def mT(self) -> Tensor:  # noqa: torch-like API
        return self.transpose(-2, -1)

    @__view
    def permute(self, dims: Size) -> Tensor:
        if self.ndim != len(dims):
            raise RuntimeError("Number of dimensions in the tensor input does not match "
                               "the length of the desired ordering of dimensions i.e. "
                               f"input.dim() = {self.ndim} is not equal to len(dims) = {len(dims)}.")
        else:
            check_dims(dims, self.ndim)

        return func._permute(self, dims)

    @__view
    def squeeze(self, dim: int | Size | None = None) -> Tensor:
        check_dims(dim, self.ndim)
        return func._squeeze(self, dim)

    @__view
    def unsqueeze(self, dim: int | Size) -> Tensor:
        check_dims(dim, self.ndim + len(flatten(dim)))
        return func._unsqueeze(self, dim)

    @__view
    def expand(self, *sizes: int | Size) -> Tensor:
        # TODO: exceptions

        return func._expand(self, sizes)

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

    def nelement(self) -> int:
        return self._data.size

    def detach(self):  # unlike torch makes a full copy of a tensor
        return tensor(np.copy(self._data))

    def retain_grad(self) -> None:
        if not self._requires_grad:
            raise RuntimeError("Can't retain_grad on Tensor that has requires_grad=False.")

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


def rand(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.random.rand(*size), dtype, requires_grad)


def randn(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.random.randn(*size), dtype, requires_grad)


def ones(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.ones(size, dtype), requires_grad=requires_grad)


def ones_like(input: Tensor,  # noqa: torch-like API
              dtype: type | np.dtype | None = None,
              requires_grad: bool = False) -> Tensor:
    return tensor(np.ones(input.shape, dtype), requires_grad=requires_grad)


def zeros(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.zeros(size, dtype), requires_grad=requires_grad)


def zeros_like(input: Tensor,  # noqa: torch-like API
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.zeros(input.shape, dtype), requires_grad=requires_grad)


Operand = Scalar | np.ndarray | Tensor
tensor = Tensor
