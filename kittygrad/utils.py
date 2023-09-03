from .constants import *


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(x: list) -> list:
    return sum(map(flatten, x), []) if isinstance(x, list) else [x]


def inv_permutation(permutation:  Size) -> Size:
    if not permutation:
        return tuple()

    permutation = np.array(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv.tolist()


def check_dim(ndim: int, dim: int) -> None:
    min_dim, max_dim = sorted([-ndim, ndim - 1])
    if not min_dim <= dim <= max_dim:
        raise IndexError("Dimension out of range (expected to be in range of [{}, {}], but got {})."
                         .format(min_dim, max_dim, dim))

