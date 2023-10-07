from __future__ import annotations

import warnings
from contextlib import nullcontext
from functools import wraps

from .engine import BackwardGraph
from ..utils import *


class no_grad:  # noqa: torch-like api
    _instance = None

    def __new__(cls, orig_func: typing.Callable | None = None) -> typing.Callable | no_grad | nullcontext:
        if orig_func is not None:
            @wraps(orig_func)
            def decorated(*args, **kwargs) -> typing.Any:
                with cls():
                    return orig_func(*args, **kwargs)
            return decorated

        elif cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance

        else:
            warnings.warn("Calling no_grad more then once has no additional effect.")
            return nullcontext()

    def __enter__(self) -> None:
        BackwardGraph.disabled = True

    def __exit__(self, *_args, **_kwargs) -> None:
        BackwardGraph.disabled = False
        no_grad._instance = None


class Function:
    pass
