from kittygrad.core import Size


def dim2tuple(dim: int | Size | None, ndim: int) -> Size:
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
