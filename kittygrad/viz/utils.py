import typing


def truncated_graph_error() -> typing.NoReturn:
    raise RuntimeError("Visualization of the computational graph "
                       "must be built starting from the leaves.")


def obj_name(obj: typing.Any) -> str:
    return str(id(obj))
