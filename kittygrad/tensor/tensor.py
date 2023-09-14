from __future__ import annotations

from ctypes import c_int64 as mutable_int
from functools import wraps
import warnings

import kittygrad.func as func
from ..autograd.engine import FnBackward, check_locks
from ..utils import *


class Tensor:
    def __init__(self, data, dtype: type | np.dtype | None = None, requires_grad: bool = False) -> None:
        if type(data) == type(self):
            raise RuntimeError("If you want to create a new tensor from another, use "
                               "sourceTensor.detach() and then specify the requires_grad attribute.")
        elif isinstance(data, np.generic):
            data = np.array(data)

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
                warnings.warn("Passed NumPy array has an unsupported data type. "
                              f"Created a copy based on '{DEFAULT_DTYPE.__name__}' dtype.")
            self._data = np.array(data, DEFAULT_DTYPE)
        elif not is_ndarray and supported_dtype:
            self._data = np.array(data, dtype)
        else:
            self._data = data.astype(dtype, copy=False)

        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None  # points to a node in a backward graph
        self._is_leaf = True
        self._retains_grad = False
        self._version = mutable_int(0)

    # ============================================= Tensor Representation ==============================================

    def __repr__(self) -> str:
        suffix = ''
        if self.dtype != DEFAULT_DTYPE:
            suffix += f', dtype={self.dtype}'
        if self._grad_fn is not None:
            suffix += f', grad_fn={self._grad_fn}'
        elif self._requires_grad:
            suffix += ', requires_grad=True'
        suffix += ')'

        return REPR_PREFIX + np.array2string(
            a=self._data,
            precision=REPR_PRECISION,
            separator=REPR_SEPARATOR,
            prefix=REPR_PREFIX,
            suffix=suffix,
        ) + suffix

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
            grad_type_error()
        elif new_grad.shape != self.shape:
            grad_shape_error()
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
    def __operator_handler(op_symbol: str, reverse: bool = False, broadcast: bool = True) -> typing.Callable:
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
                    if broadcast:
                        self, other = func.broadcast_tensors(self, other)
                    return operator(self, other)

                raise TypeError("Unsupported operand type(s) for {}: '{}' and '{}'."
                                .format(op_symbol, type(first_operand).__name__, type(second_operand).__name__))

            return handler
        return handler_decor

    @staticmethod
    def __inplace_operation(method: typing.Callable) -> typing.Callable:  # TODO: test
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> Tensor:
            if self._is_leaf and self._requires_grad:
                raise RuntimeError("A leaf Variable that requires grad is being used in an in-place operation.")
            elif not self._data.flags['WRITEABLE']:
                raise RuntimeError("The inplace operation cannot be applied to a read-only tensor. If this "
                                   "tensor is a view of another, you can try to do the same operation with it.")

            out = method(self, *args, **kwargs)
            self._version.value += 1
            return out

        return wrapper

    @staticmethod
    def __view(method: typing.Callable) -> typing.Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> Tensor:
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

    # TODO: std, abs

    @__operator_handler(op_symbol='@', broadcast=False)
    def __matmul__(self, other: np.ndarray | Tensor) -> Tensor:
        return func.matmul(self, other)

    @__operator_handler(op_symbol='@', reverse=True, broadcast=False)
    def __rmatmul__(self, other: Tensor) -> Tensor:  # TODO: other is only a tensor (for now)
        return func.matmul(other, self)

    # ================================================== Inplace Func ==================================================

    @__inplace_operation
    @__operator_handler(op_symbol='+=')
    def __iadd__(self, other: Operand) -> Tensor:
        return func._iadd(self, other)

    @__inplace_operation
    @__operator_handler(op_symbol='-=')
    def __isub__(self, other: Operand) -> Tensor:
        return func._isub(self, other)

    @__inplace_operation
    @__operator_handler(op_symbol='*=')
    def __imul__(self, other: Operand) -> Tensor:
        return func._imul(self, other)

    @__inplace_operation
    @__operator_handler(op_symbol='/=')
    def __itruediv__(self, other: Operand) -> Tensor:
        return func._idiv(self, other)

    @__inplace_operation
    @__operator_handler(op_symbol='**=')
    def __ipow__(self, other: Operand) -> Tensor:
        return func._ipow(self, other)

    def __imatmul__(self, other: np.ndarray | Tensor) -> typing.NoReturn:
        raise NotImplementedError("Inplace matrix multiplication is not implemented.")

    # ====================================================== View ======================================================

    @__view
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return func._transpose(self, dim0, dim1)

    @property
    def mT(self) -> Tensor:  # noqa: torch-like API
        return self.transpose(-2, -1)

    @__view
    def permute(self, dims: Size) -> Tensor:
        return func._permute(self, dims)

    @__view
    def squeeze(self, dim: int | Size | None = None) -> Tensor:
        return func._squeeze(self, dim)

    @__view
    def unsqueeze(self, dim: int | Size) -> Tensor:
        return func._unsqueeze(self, dim)

    @__view
    def expand(self, *sizes: int) -> Tensor:
        return func.broadcast_to(self, sizes)

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

    def type(self, dtype: type | np.dtype) -> Tensor:
        if dtype != self.dtype:
            return func._type(self, dtype)
        else:
            return self

    def nelement(self) -> int:
        return self._data.size

    def detach(self) -> Tensor:  # unlike torch makes a full copy of a tensor
        return tensor(self._data.copy())

    def retain_grad(self) -> None:
        if not self._requires_grad:
            raise RuntimeError("Can't retain_grad on Tensor that has requires_grad=False.")

        if not self._is_leaf:
            self._retains_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        # ENTRY POINT
        if not self._requires_grad:
            raise RuntimeError("Tensor does not require grad and does not have a grad_fn.")
        elif gradient is None and self.ndim:
            raise RuntimeError("Grad can be implicitly created only for scalar outputs.")

        if gradient is None:
            gradient = tensor(1, dtype=self.dtype)

        if self._is_leaf:
            self.grad = gradient
            return

        if self._grad_fn is None:
            redundant_backward_error()
        elif self.shape != gradient.shape:
            grad_shape_error()
        elif self.dtype != gradient.dtype:
            grad_type_error()
        elif self._grad_fn._lock != 0:
            warnings.warn("A .backward() call from the middle of the computational graph was noticed.")

        # ENGINE
        self._grad_fn._lock = 1
        temp = self._grad_fn

        self._grad_fn.propagate(gradient._data)
        check_locks(temp)


Operand = Scalar | np.ndarray | Tensor
tensor = Tensor
