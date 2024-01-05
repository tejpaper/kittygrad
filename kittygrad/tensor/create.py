import numpy as np

from kittygrad.tensor.tensor import Tensor, tensor
from kittygrad.utils.constants import Size, DEFAULT_DTYPE


def rand(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    """
    Create a tensor with random values drawn from a uniform distribution.

    Parameters
    ----------
    *size : Size
        The size of the tensor along each dimension.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor with random values drawn from a uniform distribution.
    """
    return tensor(
        data=np.random.rand(*size),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)


def randn(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    """
    Create a tensor with random values drawn from a standard normal distribution.

    Parameters
    ----------
    *size : Size
        The size of the tensor along each dimension.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor with random values drawn from a standard normal distribution.
    """
    return tensor(
        data=np.random.randn(*size),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)


def ones(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with ones.

    Parameters
    ----------
    *size : Size
        The size of the tensor along each dimension.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor filled with ones.
    """
    return tensor(
        data=np.ones(size, dtype=DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def ones_like(input: Tensor,  # noqa: torch-like API
              dtype: type | np.dtype | None = None,
              requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with ones, having the same shape as the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor whose shape will be used for the new tensor.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor filled with ones and having the same shape as the input tensor.
    """
    return tensor(
        data=np.ones(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)


def zeros(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with zeros.

    Parameters
    ----------
    *size : Size
        The size of the tensor along each dimension.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor filled with zeros.
    """
    return tensor(
        data=np.zeros(size, dtype=DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def zeros_like(input: Tensor,  # noqa: torch-like API
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with zeros, having the same shape as the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor whose shape will be used for the new tensor.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor filled with zeros and having the same shape as the input tensor.
    """
    return tensor(
        data=np.zeros(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)


def empty(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    """
    Create an empty tensor with uninitialized values.

    Parameters
    ----------
    *size : Size
        The size of the tensor along each dimension.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor with uninitialized values.
    """
    return tensor(
        data=np.empty(size, dtype=DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def empty_like(input: Tensor,  # noqa: torch-like API
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    """
    Create an empty tensor with uninitialized values, having the same shape as the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor whose shape will be used for the new tensor.

    dtype : type or np.dtype or None, optional
        The desired data type for the tensor. If None, uses the default data type.

    requires_grad : bool, optional
        If True, the tensor will be marked for gradient computation.

    Returns
    -------
    Tensor
        A new tensor with uninitialized values and having the same shape as the input tensor.
    """
    return tensor(
        data=np.empty(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)
