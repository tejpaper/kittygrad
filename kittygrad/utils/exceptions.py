import typing


def inplace_modification_error() -> typing.NoReturn:
    raise RuntimeError("One of the variables needed for gradient computation "
                       "has been modified by an inplace operation.")


def redundant_backward_error() -> typing.NoReturn:
    raise RuntimeError("Trying to backward through the graph a second time.")


def truncated_graph_error() -> typing.NoReturn:
    raise RuntimeError("Visualization of the computational graph "
                       "must be built starting from the leaves.")
