from .tensor import *


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
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return tensor(np.ones(input.shape), dtype, requires_grad=requires_grad)


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
