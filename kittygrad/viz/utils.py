import typing


def obj_name(obj: typing.Any) -> str:
    return str(id(obj))
