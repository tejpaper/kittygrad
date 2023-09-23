import numpy as np

from .tensor import *


def rand(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.random.rand(*size),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad
    )


def randn(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.random.randn(*size),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad
    )


def ones(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.ones(size, dtype=DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad
    )


def ones_like(input: Tensor,  # noqa: torch-like API
              dtype: type | np.dtype | None = None,
              requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.ones(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad
    )


def zeros(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.zeros(size, dtype=DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad
    )


def zeros_like(input: Tensor,  # noqa: torch-like API
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    return tensor(
        data=np.zeros(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad
    )
