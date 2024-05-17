from __future__ import annotations

import inspect
import sys
import typing

T = typing.TypeVar('T')


class Annotation(str):
    def __new__(cls, content: T) -> T | Annotation:
        if isinstance(content, str):
            return super().__new__(cls, content)
        else:
            return content

    def __repr__(self) -> str:
        return super().__str__()


def prepare_namespace(module_name: str) -> list[str]:
    namespace, members = map(list, zip(*inspect.getmembers(
        sys.modules[module_name],
        lambda x: hasattr(x, '__module__') and x.__module__.startswith(module_name),
    )))

    while members:
        obj = members.pop()
        signature = inspect.signature(obj)

        # TODO: remove me after a bunch of tests
        assert inspect.isclass(obj) or inspect.isfunction(obj)

        obj.__module__ = module_name
        obj.__signature__ = signature.replace(
            parameters=[param.replace(annotation=Annotation(param.annotation))
                        for param in signature.parameters.values()],
            return_annotation=Annotation(signature.return_annotation),
        )

        if inspect.isclass(obj):
            _, methods = zip(*inspect.getmembers(obj, inspect.isfunction))
            members.extend(methods)

    return namespace
