from __future__ import annotations

import warnings
from ctypes import c_int64 as mutable_int

import kittygrad.func as func
from ..autograd.engine import FnBackward, check_locks
from ..func.handler import autocast, inplace, view
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
            raise RuntimeError(
                "You can only change requires_grad flags of leaf variables." + (
                    "" if new_value else
                    "If you want to use a computed variable in a subgraph that doesn't "
                    "require differentiation use var_no_grad = var.detach()."))

        self._requires_grad = new_value

    @property
    def grad(self) -> Tensor | None:
        if not self._is_leaf and not self._retains_grad:
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

    # ================================================== Numpy Issues ==================================================

    def __array_ufunc__(*args, **kwargs) -> typing.NoReturn:
        raise NotImplementedError("Unsupported operation with NumPy array. Try swapping operands.")  # TODO: mb develop

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

    @autocast(op_symbol='+')
    def __add__(self, other: Operand) -> Tensor:
        return func._add(self, other)

    @autocast(op_symbol='+', reverse=True)
    def __radd__(self, other: Operand) -> Tensor:
        return func._add(self, other)

    @autocast(op_symbol='-')
    def __sub__(self, other: Operand) -> Tensor:
        return func._sub(self, other)

    @autocast(op_symbol='-', reverse=True)
    def __rsub__(self, other: Operand) -> Tensor:
        return func._sub(other, self)

    @autocast(op_symbol='*')
    def __mul__(self, other: Operand) -> Tensor:
        return func._mul(self, other)

    @autocast(op_symbol='*', reverse=True)
    def __rmul__(self, other: Operand) -> Tensor:
        return func._mul(self, other)

    @autocast(op_symbol='/')
    def __truediv__(self, other: Operand) -> Tensor:
        return func._div(self, other)

    @autocast(op_symbol='/', reverse=True)
    def __rtruediv__(self, other: Operand) -> Tensor:
        return func._div(other, self)

    @autocast(op_symbol='**')
    def __pow__(self, power: Operand) -> Tensor:
        return func._pow(self, power)

    @autocast(op_symbol='**', reverse=True)
    def __rpow__(self, power: Operand) -> Tensor:
        return func._pow(power, self)

    def sum(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return func._sum(self, dim, keepdim)

    def mean(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return func._mean(self, dim, keepdim)

    # TODO: std, abs

    @autocast(op_symbol='@', broadcasting=False, prohibited_types=[Scalar])
    def __matmul__(self, other: np.ndarray | Tensor) -> Tensor:
        return func.matmul.__wrapped__(self, other)

    @autocast(op_symbol='@', reverse=True, broadcasting=False, prohibited_types=[Scalar])
    def __rmatmul__(self, other: Tensor) -> Tensor:  # TODO: other is only a tensor (for now)
        return func.matmul.__wrapped__(other, self)

    # ================================================== Inplace Func ==================================================

    @inplace(op_symbol='+=')
    def __iadd__(self, other: Operand) -> Tensor:
        return func._iadd(self, other)

    @inplace(op_symbol='-=')
    def __isub__(self, other: Operand) -> Tensor:
        return func._isub(self, other)

    @inplace(op_symbol='*=')
    def __imul__(self, other: Operand) -> Tensor:
        return func._imul(self, other)

    @inplace(op_symbol='/=')
    def __itruediv__(self, other: Operand) -> Tensor:
        return func._idiv(self, other)

    @inplace(op_symbol='**=')
    def __ipow__(self, other: Operand) -> Tensor:
        return func._ipow(self, other)

    def __imatmul__(self, other: np.ndarray | Tensor) -> typing.NoReturn:
        raise NotImplementedError("Inplace matrix multiplication is not implemented.")

    # ====================================================== View ======================================================

    @view
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return func._transpose(self, dim0, dim1)

    @property
    def mT(self) -> Tensor:  # noqa: torch-like API
        if self.ndim < 2:
            raise RuntimeError("tensor.mT is only supported on matrices or batches of "
                               f"matrices. Got {self.ndim}D tensor.")
        else:
            return self.transpose(-2, -1)

    @view
    def permute(self, dims: Size) -> Tensor:
        return func._permute(self, dims)

    @view
    def squeeze(self, dim: int | Size | None = None) -> Tensor:
        return func._squeeze(self, dim)

    @view
    def unsqueeze(self, dim: int | Size) -> Tensor:
        return func._unsqueeze(self, dim)

    def expand(self, *sizes: int) -> Tensor:
        return func.broadcast_to(self, sizes)

    def __getitem__(self, *args, **kwargs) -> Tensor:
        if self._requires_grad:
            return tensor(data=self._data.__getitem__(*args, **kwargs))  # TODO: SelectBackward
        else:
            return tensor(data=self._data.__getitem__(*args, **kwargs))

    # @inplace  TODO
    def __setitem__(self, key, value) -> None:
        if type(value) == type(self):
            value = value._data

        if self._requires_grad:
            self._data.__setitem__(key, value)  # TODO: CopySlices
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

    def item(self) -> Scalar:
        if self.nelement() != 1:
            raise RuntimeError(f"A Tensor with {self.nelement()} elements cannot be converted to Scalar.")
        else:
            return self._data.item()

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
            raise RuntimeError("Initial gradient has data of a different size.")
        elif self.dtype != gradient.dtype:
            raise TypeError("Initial gradient has data of a different type.")
        elif self._grad_fn._lock != 0:
            warnings.warn("A .backward() call from the middle of the computational graph was noticed.")

        # ENGINE
        self._grad_fn._lock = 1
        temp = self._grad_fn

        self._grad_fn.propagate(gradient._data)
        check_locks(temp)


Operand = Scalar | np.ndarray | Tensor
tensor = Tensor
