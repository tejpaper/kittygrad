from __future__ import annotations

import abc
import warnings
from contextlib import nullcontext
from functools import wraps

from inflection import underscore

from kittygrad.tensor import tensor
from .engine import FnBackward, BackwardGraph
from ..utils import *


class no_grad:
    def __new__(cls, orig_func: typing.Callable | None = None) -> typing.Callable | no_grad | nullcontext:
        if orig_func is not None:
            @wraps(orig_func)
            def decorated(*args, **kwargs) -> typing.Any:
                with cls():
                    return orig_func(*args, **kwargs)

            decorated.disables_grad = True
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
        def disable_grad(ctx: DotDict, *args, **kwargs):
            output_tensor = function(ctx, *args, **kwargs)
            output_tensor._requires_grad = False

            return output_tensor
        return disable_grad


class FunctionMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace, *, output_version_check: bool = False) -> FunctionMeta:
        if name == 'Function':
            return super().__new__(mcs, name, bases, namespace)

        forward = namespace.get('forward')
        backward = namespace.get('backward')

        for base in bases:
            if base != Function:
                if not forward:
                    forward = vars(base).get('forward')
                if not backward:
                    backward = vars(base).get('backward')

        if not forward or not backward:
            return super().__new__(mcs, name, bases, namespace)
        elif hasattr(forward, 'disables_grad'):
            raise RuntimeError("There is no point in creating a Function class with gradient flow disabled. "
                               "Use a standard function instead.")
        elif hasattr(backward, 'disables_grad'):
            warnings.warn("There is no need to explicitly disable gradient flow in the backward method. "
                          "This happens implicitly.")

        forward = BackwardGraph.disable(forward)
        backward = BackwardGraph.disable(backward)

        def _forward(ctx: DotDict, self: typing.Self[Function], *args, **kwargs) -> Tensor:
            ctx.custom_function = self
            self.ctx = ctx
            return forward(self, *args, **kwargs)

        def _backward(self: typing.Self[FnBackward]) -> None:
            custom_function = self._ctx.custom_function

            if output_version_check:
                self._inplace_modification_check()

            custom_function.ctx = self._ctx
            prev_grads = backward(custom_function, tensor(self._grad))

            if not isinstance(prev_grads, Iterable):
                prev_grads = (prev_grads,)

            for next_fn, grad in zip(self._next_functions, prev_grads):
                if grad is not None:
                    next_fn.propagate(grad._data)

        _forward.__name__ = f'_{underscore(name)}'
        backward_node_cls = type(name + 'Backward', (FnBackward,), dict(_propagate=_backward))
        namespace['__call__'] = BackwardGraph.mount(backward_node_cls)(_forward)

        return super().__new__(mcs, name, bases, namespace)


class Function(metaclass=FunctionMeta):
    def __init__(self) -> None:
        self.ctx = DotDict(saved_tensors=[])  # placeholder

    def __call__(self, *args, **kwargs) -> Tensor:
        raise RuntimeError("Impossible exception.")  # overridden by FunctionMeta

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor | tuple[Tensor | None, ...]:
        raise NotImplementedError
