from __future__ import annotations

from collections.abc import Iterable

from .constants import *
import kittygrad.tensor as tsr


manual_seed = np.random.seed


def flatten(x: tuple[int, ...] | list[int] | Scalar) -> list:
    return sum(map(flatten, x), []) if isinstance(x, Iterable) else [x]


def inv_permutation(permutation: Size) -> Size:
    if not permutation:
        return tuple()

    permutation = np.array(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv.tolist()


def check_types(tensor: tsr.Tensor, other: tsr.Tensor) -> None:
    if tensor.dtype != other.dtype:
        raise TypeError("Operands type mismatch: {} != {}."
                        .format(tensor.dtype, other.dtype))


def check_dim(dim: int, ndim: int) -> None:
    min_dim, max_dim = sorted([-ndim, ndim - 1])
    if not min_dim <= dim <= max_dim:
        raise IndexError("Dimension out of range (expected to be in range of [{}, {}], but got {})."
                         .format(min_dim, max_dim, dim))


def normalize_dims(dims: Size, ndim: int) -> Size:
    return [dim + ndim if dim < 0 else dim for dim in dims]


def check_dims(dims: int | Size | None, ndim: int) -> None:
    if dims is None:
        return
    elif isinstance(dims, int):
        dims = [dims]
    elif len(set(normalize_dims(dims, ndim))) != len(dims):
        raise RuntimeError("Duplicate dims are not allowed.")

    for dim in dims:
        check_dim(dim, ndim)
