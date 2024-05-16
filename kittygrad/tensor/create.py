from __future__ import annotations

import kittygrad.core as core
import kittygrad.tensor.tensor as tsr


def rand(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.random.rand(*size),
        dtype=core.DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)


def randn(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.random.randn(*size),
        dtype=core.DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)


def ones(*size: Size,
         dtype: type | np.dtype | None = None,
         requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.ones(size, dtype=core.DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def ones_like(input: Tensor,
              dtype: type | np.dtype | None = None,
              requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.ones(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)


def zeros(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.zeros(size, dtype=core.DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def zeros_like(input: Tensor,
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.zeros(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)


def empty(*size: Size,
          dtype: type | np.dtype | None = None,
          requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.empty(size, dtype=core.DEFAULT_DTYPE if dtype is None else dtype),
        requires_grad=requires_grad)


def empty_like(input: Tensor,
               dtype: type | np.dtype | None = None,
               requires_grad: bool = False) -> Tensor:
    return tsr.tensor(
        data=core.np.empty(input.shape, dtype=input.dtype if dtype is None else dtype),
        requires_grad=requires_grad)
