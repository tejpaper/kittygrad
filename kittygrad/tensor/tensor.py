from __future__ import annotations

import warnings
from ctypes import c_int64 as mutable_int
from types import EllipsisType, NoneType

from kittygrad.autograd.engine import FnBackward, check_locks
from kittygrad.autograd.utils import redundant_backward_error
from kittygrad.core import *
from kittygrad.func import activation, ops, view
from kittygrad.func.handler import autocast, inplace, share
from kittygrad.tensor.utils import flatten


class Tensor:  # TODO: test empty tensor cases
    def __init__(self, data, *, dtype: type | np.dtype | None = None, requires_grad: bool = False) -> None:
        if type(data) is type(self):
            raise RuntimeError("If you want to create a new tensor from another, use "
                               "source_tensor.detach() and then specify the requires_grad attribute.")
        elif isinstance(data, np.generic):
            data = np.array(data)

        is_ndarray = isinstance(data, np.ndarray)
        dtype_unknown = dtype is None

        if dtype_unknown:
            if not is_ndarray:
                if flattened_data := flatten(data):
                    dtype = np.result_type(*flatten(data))
                if dtype == np.float_ or not flattened_data:
                    dtype = DEFAULT_DTYPE
            else:
                dtype = data.dtype

        supported_dtype = dtype in ALL_DTYPES

        if not dtype_unknown and not supported_dtype:
            raise TypeError(f"Data type '{dtype.__name__}' is not supported.")
        elif dtype_unknown and not supported_dtype:
            if is_ndarray:
                warnings.warn(f"Passed NumPy array has an unsupported data type '{data.dtype}'. "
                              f"Created a copy based on '{DEFAULT_DTYPE.__name__}' dtype.")
            self._data = np.array(data, DEFAULT_DTYPE)
        elif not is_ndarray:
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
        if self.nelement() == 0:
            suffix += f', size={self.shape}'
        if self.dtype != DEFAULT_DTYPE:
            suffix += f', dtype={self.dtype}'
        if self._grad_fn is not None:
            suffix += f', grad_fn={self._grad_fn}'
        elif self._requires_grad:
            suffix += ', requires_grad=True'
        suffix += ')'

        return REPR_PREFIX + np.array2string(
            a=self._data,
            max_line_width=REPR_MAX_LINE_WIDTH,
            precision=REPR_PRECISION,
            separator=REPR_SEPARATOR,
            prefix=REPR_PREFIX,
            suffix=suffix,
            floatmode=REPR_FLOATMODE,
        ) + suffix

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:  # TODO: test
        if self.ndim == 0:
            raise TypeError("The length is not defined for a 0D tensor.")
        else:
            return self.shape[0]

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

        if type(new_grad) is not type(self):
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

    # ===================================== Compatibility with 3rd party libraries =====================================

    __array_ufunc__ = None

    def __array__(self, dtype: type | np.dtype | None = None) -> np.ndarray:  # TODO: test
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype, copy=False)

    def __iter__(self) -> typing.Iterator[Tensor]:  # TODO: test
        # resolves issue with pandas.api.types.is_list_like
        if self.ndim == 0:
            raise TypeError("Iteration over a 0D tensor is not possible.")
        else:
            return map(tensor, self._data)

    # ====================================================== Func ======================================================

    def __pos__(self) -> typing.Self:
        return self

    def __neg__(self) -> Tensor:
        return ops._neg(self)

    def __abs__(self) -> Tensor:
        return ops._abs(self)

    def abs(self) -> Tensor:
        return ops._abs(self)

    def exp(self) -> Tensor:
        return ops._exp(self)

    def log(self) -> Tensor:
        return ops._log(self)

    def sigmoid(self) -> Tensor:
        return activation._sigmoid(self)

    def tanh(self) -> Tensor:
        return activation._tanh(self)

    def relu(self) -> Tensor:
        return activation._relu(self)

    @autocast(op_symbol='+')
    def __add__(self, other: Operand) -> Tensor:
        return ops._add(self, other)

    @autocast(op_symbol='+', reverse=True)
    def __radd__(self, other: Operand) -> Tensor:
        return ops._add(self, other)

    @autocast(op_symbol='-')
    def __sub__(self, other: Operand) -> Tensor:
        return ops._sub(self, other)

    @autocast(op_symbol='-', reverse=True)
    def __rsub__(self, other: Operand) -> Tensor:
        return ops._sub(other, self)

    @autocast(op_symbol='*')
    def __mul__(self, other: Operand) -> Tensor:
        return ops._mul(self, other)

    @autocast(op_symbol='*', reverse=True)
    def __rmul__(self, other: Operand) -> Tensor:
        return ops._mul(self, other)

    @autocast(op_symbol='/')
    def __truediv__(self, other: Operand) -> Tensor:
        return ops._div(self, other)

    @autocast(op_symbol='/', reverse=True)
    def __rtruediv__(self, other: Operand) -> Tensor:
        return ops._div(other, self)

    @autocast(op_symbol='**')
    def __pow__(self, power: Operand) -> Tensor:
        return ops._pow(self, power)

    @autocast(op_symbol='**', reverse=True)
    def __rpow__(self, power: Operand) -> Tensor:
        return ops._pow(power, self)

    def sum(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return ops._sum(self, dim, keepdim)

    def mean(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        return ops._mean(self, dim, keepdim)

    def var(self, dim: int | Size | None = None, correction: int = 1, keepdim: bool = False) -> Tensor:
        return ops._var(self, dim, correction, keepdim)

    def std(self, dim: int | Size | None = None, correction: int = 1, keepdim: bool = False) -> Tensor:
        return ops._std(self, dim, correction, keepdim)

    @autocast(op_symbol='@', broadcasting=False, prohibited_types=[Scalar])
    def __matmul__(self, other: np.ndarray | Tensor) -> Tensor:
        return ops.matmul.__wrapped__(self, other)

    @autocast(op_symbol='@', reverse=True, broadcasting=False, prohibited_types=[Scalar])
    def __rmatmul__(self, other: np.ndarray | Tensor) -> Tensor:
        return ops.matmul.__wrapped__(other, self)

    # ================================================== Inplace Func ==================================================

    @inplace(op_symbol='+=')
    def __iadd__(self, other: Operand) -> Tensor:
        return ops._iadd(self, other)

    @inplace(op_symbol='-=')
    def __isub__(self, other: Operand) -> Tensor:
        return ops._isub(self, other)

    @inplace(op_symbol='*=')
    def __imul__(self, other: Operand) -> Tensor:
        return ops._imul(self, other)

    @inplace(op_symbol='/=')
    def __itruediv__(self, other: Operand) -> Tensor:
        return ops._idiv(self, other)

    @inplace(op_symbol='**=')
    def __ipow__(self, other: Operand) -> Tensor:
        return ops._ipow(self, other)

    def __imatmul__(self, other: np.ndarray | Tensor) -> typing.NoReturn:
        raise NotImplementedError("Inplace matrix multiplication is not implemented.")

    # ====================================================== View ======================================================

    @share
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return view._transpose(self, dim0, dim1)

    @property
    def mT(self) -> Tensor:  # noqa: torch-like API
        if self.ndim < 2:
            raise RuntimeError("tensor.mT is only supported on matrices or batches of "
                               f"matrices. Got {self.ndim}D tensor.")
        else:
            return self.transpose(-2, -1)

    @share
    def permute(self, dims: Size) -> Tensor:
        return view._permute(self, dims)

    @share
    def squeeze(self, dim: int | Size | None = None) -> Tensor:
        return view._squeeze(self, dim)

    @share
    def unsqueeze(self, dim: int | Size) -> Tensor:
        return view._unsqueeze(self, dim)

    def expand(self, *sizes: int) -> Tensor:
        return view.broadcast_to(self, sizes)

    def __getitem__(self, key) -> Tensor:
        if isinstance(key, tuple):
            for ind in key:
                if type(ind) not in (int, slice, EllipsisType, NoneType):
                    return view._index(self, key)

        elif isinstance(key, list | np.ndarray):
            return view._index(self, key)

        return view._index_view(self, key)

    def __setitem__(self, key, value: Operand) -> None:
        return view._index_put(self, value, key)

    # =================================================== Comparison ===================================================

    def __lt__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data < other

    def __gt__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data > other

    def __le__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data <= other

    def __ge__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data >= other

    def __eq__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data == other

    def __ne__(self, other: Operand) -> np.ndarray:
        if type(other) is type(self):
            other = other._data
        return self._data != other

    # ================================================== Interaction ===================================================

    def type(self, dtype: type | np.dtype) -> Tensor | typing.Self:
        if dtype != self.dtype:
            return ops._type(self, dtype)
        else:
            return self

    def nelement(self) -> int:
        return self._data.size

    def item(self) -> Scalar:
        if self.nelement() != 1:
            raise RuntimeError(f"A Tensor with {self.nelement()} elements cannot be converted to Scalar.")
        else:
            return self._data.item()

    def numpy(self, *, copy: bool = False) -> np.ndarray:
        if copy:
            return self._data.copy()
        else:
            return self._data

    def detach(self) -> Tensor:
        # unlike torch creates a copy that does not share the same storage with the source tensor
        return tensor(self.numpy(copy=True))

    def clone(self) -> Tensor:
        return ops._clone(self)

    def retain_grad(self) -> None:
        if not self._requires_grad:
            raise RuntimeError("Can't retain_grad on Tensor that has requires_grad=False.")

        if not self._is_leaf:
            self._retains_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        # entry point
        if not self._requires_grad:
            raise RuntimeError("Tensor does not require grad and does not have a grad_fn.")
        elif gradient is None and self.ndim:
            raise RuntimeError("Gradient can be implicitly created only for scalar outputs.")

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

        # engine
        self._grad_fn._lock = 1
        grad_fn = self._grad_fn

        self._grad_fn.propagate(gradient._data)
        check_locks(grad_fn)


Operand = Scalar | np.ndarray | Tensor
tensor = Tensor
