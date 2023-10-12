from __future__ import annotations

import abc
import warnings
from contextlib import nullcontext
from functools import wraps

from kittygrad.tensor import Tensor
from .engine import FnBackward, BackwardGraph
from ..utils import *


class no_grad:
    def __new__(cls, orig_func: typing.Callable | None = None) -> typing.Callable | no_grad | nullcontext:
        if orig_func is not None:
            @wraps(orig_func)
            def decorated(*args, **kwargs) -> typing.Any:
                with cls():
                    return orig_func(*args, **kwargs)
            return decorated

        elif BackwardGraph.pre_builder_hooks.no_grad is None:
            return super().__new__(cls)

        else:
            warnings.warn("Calling no_grad more then once has no additional effect.")
            return nullcontext()

    def __enter__(self) -> None:
        BackwardGraph.pre_builder_hooks.no_grad = self._hook

    def __exit__(self, *_args, **_kwargs) -> None:
        del BackwardGraph.pre_builder_hooks.no_grad

    @staticmethod
    def _hook(function: typing.Callable) -> typing.Callable:
        @wraps(function)
        def disable_grad(*args):
            output_tensor = function(*args)
            output_tensor._requires_grad = False

            return output_tensor
        return disable_grad


class FunctionMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        if name != 'Function':
            forward = BackwardGraph.disable(namespace['forward'])
            backward = BackwardGraph.disable(namespace['backward'])

            def _forward(self, *args) -> Tensor:
                *args, ctx = args
                out = forward(self, *args)
                ctx |= self.ctx
                return out

            _forward.__name__ = f'_{camel2snake(name)}'

            def _backward(self) -> None:
                prev_grads = backward(self, self._grad)
                if not isinstance(prev_grads, Iterable):
                    prev_grads = (prev_grads,)

                for next_fn, grad in zip(self._next_functions, prev_grads):
                    if grad is not None:
                        next_fn.propagate(grad._data)

            backward_node_cls = type(
                name + 'Backward',
                (FnBackward,), {
                    'ctx': property(lambda self: self._ctx),
                    '_propagate': _backward
                })

            namespace['__call__'] = BackwardGraph.mount(backward_node_cls)(_forward)

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class Function(metaclass=FunctionMeta):
    def __init__(self) -> None:
        # TODO: docs: don't use the "out" key in the context variable
        self.ctx = DotDict(saved_tensors=[])  # TODO: problems with inplace operations?

    def __call__(self, *args, **kwargs) -> Tensor:
        raise RuntimeError("Impossible exception.")  # overridden by FunctionMeta

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor | tuple[Tensor | None, ...]:
        # TODO: no inplace gradient modifications reminder
        raise NotImplementedError
