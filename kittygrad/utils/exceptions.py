import typing


def grad_shape_error() -> typing.NoReturn:
    raise RuntimeError("Assigned grad has data of a different size.")


def grad_type_error() -> typing.NoReturn:
    raise TypeError("Assigned grad has data of a different type.")


def inplace_modification_error() -> typing.NoReturn:
    raise RuntimeError("One of the variables needed for gradient computation "
                       "has been modified by an inplace operation.")


def redundant_backward_error() -> typing.NoReturn:
    raise RuntimeError("Trying to backward through the graph a second time.")
