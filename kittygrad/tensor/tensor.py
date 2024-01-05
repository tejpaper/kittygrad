from __future__ import annotations

import typing
import warnings
from ctypes import c_int64 as mutable_int
from types import EllipsisType, NoneType

from kittygrad.autograd.engine import FnBackward, check_locks
from kittygrad.func import activation, ops, view
from kittygrad.func.handler import autocast, inplace, share
from kittygrad.utils.constants import *
from kittygrad.utils.exceptions import redundant_backward_error
from kittygrad.utils.functions import flatten


class Tensor:
    """
    Multidimensional array with optional gradient support.
    """

    def __init__(self, data, *, dtype: type | np.dtype | None = None, requires_grad: bool = False) -> None:
        """
        Initialize a new Tensor.

        Parameters
        ----------
        data : array_like
            The input data for creating the tensor.

        dtype : {type, np.dtype, None}, optional
            Data type of the tensor. If None, uses the default data type.

        requires_grad : bool, optional
            If True, the tensor will track operations for gradient computation.
        """
        if type(data) is type(self):
            raise RuntimeError("If you want to create a new tensor from another, use "
                               "source_tensor.detach() and then specify the requires_grad attribute.")
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

    __array_ufunc__ = None

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
            max_line_width=REPR_MAX_LINE_WIDTH,
            precision=REPR_PRECISION,
            separator=REPR_SEPARATOR,
            prefix=REPR_PREFIX,
            suffix=suffix,
            floatmode=REPR_FLOATMODE,
        ) + suffix

    def __str__(self) -> str:
        return repr(self)

    # ============================================== Getters and Setters ===============================================

    @property  # not writable
    def data(self) -> Tensor:
        return tensor(self._data)

    @property  # not writable
    def shape(self) -> Size:
        """
        Property representing the shape of the tensor.

        Returns
        -------
        Size
            A Size object representing the dimensions of the tensor.
        """
        return self._data.shape

    @property  # not writable
    def ndim(self) -> int:
        """
        Property representing the number of dimensions (axes) of the tensor.

        Returns
        -------
        int
            The number of dimensions in the tensor.
        """
        return len(self._data.shape)

    @property  # not writable
    def dtype(self) -> np.dtype:
        """
        Property representing the data type of the tensor.

        Returns
        -------
        np.dtype
            The NumPy data type of the tensor.
        """
        return self._data.dtype

    @property
    def requires_grad(self) -> bool:
        """
        Property indicating whether the tensor requires gradient computation.

        Returns
        -------
        bool
            True if the tensor requires gradient computation, False otherwise.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, new_value: bool) -> None:
        """
        Setter method for the requires_grad property.

        Parameters
        ----------
        new_value : bool
            The new value to set for the requires_grad property.

        Raises
        ------
        RuntimeError
            If the tensor is not a leaf variable, indicating that requires_grad
            flags can only be changed for leaf variables.
        """
        if not self._is_leaf:
            raise RuntimeError(
                "You can only change requires_grad flags of leaf variables." + (
                    "" if new_value else
                    "If you want to use a computed variable in a subgraph that doesn't "
                    "require differentiation use var_no_grad = var.detach()."))

        self._requires_grad = new_value

    @property
    def grad(self) -> Tensor | None:
        """
        Property representing the gradient of the tensor.

        Returns
        -------
        Tensor or None
            The gradient tensor if available, or None if the tensor has no gradient.
        """
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
        """
        Setter method for the grad property.

        Parameters
        ----------
        new_grad : Tensor or None
            The new gradient tensor to set for the tensor.
        """
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
        """
        Property representing the backward node associated with the tensor.

        Returns
        -------
        FnBackward
            The backward node responsible for computing the gradient of the tensor.
        """
        return self._grad_fn

    @property  # not writable
    def is_leaf(self) -> bool:
        """
        Property indicating whether the tensor is a leaf variable.

        Returns
        -------
        bool
            True if the tensor is a leaf variable, False otherwise.
        """
        return self._is_leaf

    @property  # one-way writable with tensor.retain_grad()
    def retains_grad(self) -> bool:
        """
        Property indicating whether the tensor retains its gradient.

        Returns
        -------
        bool
            True if the tensor retains its gradient, False otherwise.

        Notes
        -----
        The returned value is one-way writable using `tensor.retain_grad()`.
        """
        return self._retains_grad

    @property  # not writable
    def version(self) -> int:
        """
        Property representing the version of the tensor.

        Returns
        -------
        int
            The version number associated with the tensor.
        """
        return self._version.value

    # ====================================================== Func ======================================================

    def __pos__(self) -> Tensor:
        return self

    def __neg__(self) -> Tensor:
        return ops._neg(self)

    def __abs__(self) -> Tensor:
        return ops._abs(self)

    def abs(self) -> Tensor:
        """
        Compute the element-wise absolute value of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the absolute values of the elements.
        """
        return ops._abs(self)

    def exp(self) -> Tensor:
        """
        Compute the element-wise exponential function of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the exponential values of the elements.
        """
        return ops._exp(self)

    def log(self) -> Tensor:
        """
        Compute the element-wise natural logarithm of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the natural logarithms of the elements.
        """
        return ops._log(self)

    def sigmoid(self) -> Tensor:
        """
        Compute the element-wise sigmoid function of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the sigmoid values of the elements.
        """
        return activation._sigmoid(self)

    def tanh(self) -> Tensor:
        """
        Compute the element-wise hyperbolic tangent function of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the hyperbolic tangent values of the elements.
        """
        return activation._tanh(self)

    def relu(self) -> Tensor:
        """
        Compute the element-wise rectified linear unit (ReLU) function of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the ReLU values of the elements.
        """
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
        """
        Compute the sum of tensor elements along a specified dimension.

        Parameters
        ----------
        dim : int, Size, or None, optional
            The dimension along which to compute the sum. If None, the sum is calculated over all elements.
            If an int or Size is provided, the sum is computed along the specified dimension(s).

        keepdim : bool, optional
            If True, the specified dimension(s) are retained in the result as dimensions with size 1.

        Returns
        -------
        Tensor
            A new tensor containing the sum of elements along the specified dimension(s).
        """
        return ops._sum(self, dim, keepdim)

    def mean(self, dim: int | Size | None = None, keepdim: bool = False) -> Tensor:
        """
        Compute the mean of tensor elements along a specified dimension.

        Parameters
        ----------
        dim : int, Size, or None, optional
            The dimension along which to compute the mean. If None, the mean is calculated over all elements.
            If an int or Size is provided, the mean is computed along the specified dimension(s).

        keepdim : bool, optional
            If True, the specified dimension(s) are retained in the result as dimensions with size 1.

        Returns
        -------
        Tensor
            A new tensor containing the mean of elements along the specified dimension(s).
        """
        return ops._mean(self, dim, keepdim)

    def var(self, dim: int | Size | None = None, correction: int = 1, keepdim: bool = False) -> Tensor:
        """
        Compute the variance of tensor elements along a specified dimension.

        Parameters
        ----------
        dim : int, Size, or None, optional
            The dimension along which to compute the variance. If None, the variance is calculated over all elements.
            If an int or Size is provided, the variance is computed along the specified dimension(s).

        correction : int, optional
            The degree of freedom correction to apply to the variance. Default is 1.

        keepdim : bool, optional
            If True, the specified dimension(s) are retained in the result as dimensions with size 1.

        Returns
        -------
        Tensor
            A new tensor containing the variance of elements along the specified dimension(s).
        """
        return ops._var(self, dim, correction, keepdim)

    def std(self, dim: int | Size | None = None, correction: int = 1, keepdim: bool = False) -> Tensor:
        """
        Compute the standard deviation of tensor elements along a specified dimension.

        Parameters
        ----------
        dim : int, Size, or None, optional
            The dimension along which to compute the standard deviation. If None, the standard deviation is
            calculated over all elements. If an int or Size is provided, the standard deviation is computed
            along the specified dimension(s).

        correction : int, optional
            The degree of freedom correction to apply to the standard deviation. Default is 1.

        keepdim : bool, optional
            If True, the specified dimension(s) are retained in the result as dimensions with size 1.

        Returns
        -------
        Tensor
            A new tensor containing the standard deviation of elements along the specified dimension(s).
        """
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
        """
        Transpose the tensor along the specified dimensions.

        Parameters
        ----------
        dim0 : int
            The first dimension to be transposed.

        dim1 : int
            The second dimension to be transposed.

        Returns
        -------
        Tensor
            A shared tensor with the specified dimensions transposed.
        """
        return view._transpose(self, dim0, dim1)

    @property
    def mT(self) -> Tensor:  # noqa: torch-like API
        """
        Property representing the transpose of the matrix-form tensor.

        Returns
        -------
        Tensor
            A shared tensor with dimensions transposed.
        """
        if self.ndim < 2:
            raise RuntimeError("tensor.mT is only supported on matrices or batches of "
                               f"matrices. Got {self.ndim}D tensor.")
        else:
            return self.transpose(-2, -1)

    @share
    def permute(self, dims: Size) -> Tensor:
        """
        Permute the dimensions of the tensor according to the specified order.

        Parameters
        ----------
        dims : Size
            A tuple or list representing the desired order of dimensions.

        Returns
        -------
        Tensor
            A shared tensor with dimensions permuted according to the specified order.
        """
        return view._permute(self, dims)

    @share
    def squeeze(self, dim: int | Size | None = None) -> Tensor:
        """
        Remove single-dimensional entries from the shape of the tensor.

        Parameters
        ----------
        dim : int, Size, or None, optional
            If specified, only removes the dimensions with size 1 along the specified dimension(s).
            If None, removes all dimensions with size 1.

        Returns
        -------
        Tensor
            A shared tensor with removed single-dimensional entries.
        """
        return view._squeeze(self, dim)

    @share
    def unsqueeze(self, dim: int | Size) -> Tensor:
        """
        Add a singleton dimension to the tensor at the specified position.

        Parameters
        ----------
        dim : int or Size
            The dimension index at which to add a singleton dimension. It can also be a Size object
            representing the dimensions to be added.

        Returns
        -------
        Tensor
            A shared tensor with an additional singleton dimension.
        """
        return view._unsqueeze(self, dim)

    def expand(self, *sizes: int) -> Tensor:
        """
        Expand the dimensions of the tensor by broadcasting its shape to the specified sizes.

        Parameters
        ----------
        *sizes : int
            The target sizes to which the tensor's shape will be broadcasted. Use -1 to infer the size
            for a dimension based on the original size.

        Returns
        -------
        Tensor
            A shared tensor with expanded dimensions.
        """
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

    def type(self, dtype: type | np.dtype) -> Tensor:
        """
        Change the data type of the tensor.

        Parameters
        ----------
        dtype : type or np.dtype
            The desired data type for the tensor.

        Returns
        -------
        Tensor
            A new tensor with the specified data type.
        """
        if dtype != self.dtype:
            return ops._type(self, dtype)
        else:
            return self

    def nelement(self) -> int:
        """
        Return the total number of elements in the tensor.

        Returns
        -------
        int
            The total number of elements in the tensor.
        """
        return self._data.size

    def item(self) -> Scalar:
        """
        Get the scalar value from a tensor with a single element.

        Returns
        -------
        Scalar
            The scalar value extracted from the tensor.

        Raises
        ------
        RuntimeError
            If the tensor has more than one element.
        """
        if self.nelement() != 1:
            raise RuntimeError(f"A Tensor with {self.nelement()} elements cannot be converted to Scalar.")
        else:
            return self._data.item()

    def numpy(self, *, copy: bool = False) -> np.ndarray:
        """
        Convert the tensor to a NumPy array.

        Parameters
        ----------
        copy : bool, optional
            If True, return a copy of the data. If False, return an array with shared data.

        Returns
        -------
        np.ndarray
            A NumPy array representing the tensor data.
        """
        if copy:
            return self._data.copy()
        else:
            return self._data

    def detach(self) -> Tensor:
        """
        Detach the tensor from the computation graph.

        Returns
        -------
        Tensor
            A new tensor that does not share the same storage with the source tensor.
        """
        # unlike torch creates a copy that does not share the same storage with the source tensor
        return tensor(self.numpy(copy=True))

    def clone(self) -> Tensor:
        """
        Perform a differentiable deep copy of the tensor.

        Returns
        -------
        Tensor
            A new tensor with the same data as the original tensor.
            If the source tensor has `requires_grad` activated, the resulting tensor is not a leaf
            and is part of the computational graph.
        """
        return ops._clone(self)

    def retain_grad(self) -> None:
        """
        Retain the gradient computation for the tensor.

        Notes
        -----
        This method is used to indicate that the gradient should be computed for this tensor
        during backpropagation, even if it is not a leaf in the computation graph.

        Raises
        ------
        RuntimeError
            If the tensor does not have `requires_grad` set to True.
        """

        if not self._requires_grad:
            raise RuntimeError("Can't retain_grad on Tensor that has requires_grad=False.")

        if not self._is_leaf:
            self._retains_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        """
        Perform backpropagation to compute gradients with respect to this tensor.

        Parameters
        ----------
        gradient : Tensor or None, optional
            Gradient to start with. If None, assumes the gradient is 1.

        Notes
        -----
        This method triggers the backpropagation algorithm to compute gradients for this tensor
        and its ancestors in the computation graph.
        """
        # entry point
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

        # engine
        self._grad_fn._lock = 1
        grad_fn = self._grad_fn

        self._grad_fn.propagate(gradient._data)
        check_locks(grad_fn)


Operand = Scalar | np.ndarray | Tensor
tensor = Tensor
