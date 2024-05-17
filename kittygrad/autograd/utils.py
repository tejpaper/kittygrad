from kittygrad.core import *


def inplace_modification_error() -> typing.NoReturn:
    raise RuntimeError("One of the variables needed for gradient computation "
                       "has been modified by an inplace operation.")


def redundant_backward_error() -> typing.NoReturn:
    raise RuntimeError("Trying to backward through the graph a second time.")


def inv_permutation(permutation: Size) -> Size:
    if not permutation:
        return tuple()

    permutation = np.array(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv.tolist()
