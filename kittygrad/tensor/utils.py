import typing
from collections.abc import Iterable


def flatten(x: typing.Any) -> list:
    return sum(map(flatten, x), []) if isinstance(x, Iterable) else [x]
