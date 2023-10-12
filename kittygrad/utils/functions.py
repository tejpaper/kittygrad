from __future__ import annotations

import re
import typing
from collections.abc import Iterable

from .constants import *

manual_seed = np.random.seed


def flatten(x: typing.Any) -> list:
    return sum(map(flatten, x), []) if isinstance(x, Iterable) else [x]


def camel2snake(name: str) -> str:
    """https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case"""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def inv_permutation(permutation: Size) -> Size:
    if not permutation:
        return tuple()

    permutation = np.array(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv.tolist()


def dim2tuple(dim: int | Size | None, ndim: int) -> tuple[int, ...]:
    if isinstance(dim, int):
        return (dim,)
    elif dim is not None:
        return tuple(dim)
    else:
        return tuple(range(ndim))


def separate_dims(shape: Size, mask: Size) -> tuple[Size, Size]:
    expanded_shape = list(shape)
    reps = [1] * len(shape)

    for dim in mask:
        expanded_shape[dim] = 1
        reps[dim] = shape[dim]

    return expanded_shape, reps


def check_dim(dim: int, ndim: int) -> None:
    min_dim, max_dim = sorted([-ndim, ndim - 1])
    if not min_dim <= dim <= max_dim:
        raise IndexError("Dimension out of range (expected to be in range of [{}, {}], but got {})."
                         .format(min_dim, max_dim, dim))


def normalize_dims(dims: Size, ndim: int) -> Size:
    return [dim + ndim if dim < 0 else dim for dim in dims]


def check_dims(dims: Size | None, ndim: int) -> None:
    if dims is None:
        return
    elif len(set(normalize_dims(dims, ndim))) != len(dims):
        raise RuntimeError("Duplicate dims are not allowed.")

    for dim in dims:
        check_dim(dim, ndim)
