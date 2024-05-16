from __future__ import annotations

import abc
import typing
import warnings
from contextlib import nullcontext
from functools import wraps

from inflection import underscore  # TODO: remove redundant dependency (requirements/kittygrad.txt)

from kittygrad.autograd.context import Context
from kittygrad.tensor.tensor import Tensor, tensor
from kittygrad.autograd.engine import FnBackward, BackwardGraph
from kittygrad.func.handler import normalize_args


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
        def disable_grad(ctx: Context, *args, **kwargs):
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
            raise RuntimeError("There is no point in creating a Function class with gradient "
                               "flow permanently disabled. Use a standard function instead.")
        elif hasattr(backward, 'disables_grad'):
            warnings.warn("There is no need to explicitly disable gradient flow in the backward method. "
                          "This happens implicitly.")

        backward = mcs.integrate_backward(backward, output_version_check)
        backward_node_cls = type(f'{name}Backward', (FnBackward,), dict(_propagate=backward))
        namespace['__call__'] = mcs.integrate_forward(forward, name, backward_node_cls)

        return super().__new__(mcs, name, bases, namespace)

    @staticmethod
    def integrate_forward(forward: typing.Callable, name: str,
                          backward_node_cls: typing.Type[FnBackward]) -> typing.Callable:
        forward = BackwardGraph.disable(forward)

        def integrated_forward(ctx: Context, self: typing.Self[Function], *args, **kwargs) -> Tensor:
            ctx.custom_function = self
            self.ctx = ctx
            return forward(self, *args, **kwargs)

        integrated_forward.__name__ = f'_{underscore(name)}'
        integrated_forward = BackwardGraph.mount(backward_node_cls)(integrated_forward)
        integrated_forward = normalize_args(forward)(integrated_forward)

        return integrated_forward

    @staticmethod
    def integrate_backward(backward: typing.Callable, output_version_check: bool) -> typing.Callable:
        backward = BackwardGraph.disable(backward)

        def integrated_backward(self: typing.Self[FnBackward]) -> None:
            custom_function = self._ctx.custom_function

            if output_version_check:
                self._inplace_modification_check()

            custom_function.ctx = self._ctx
            prev_grads = backward(custom_function, tensor(self._grad))

            if isinstance(prev_grads, Tensor | None):
                prev_grads = (prev_grads,)

            for next_fn, grad in zip(self._next_functions, prev_grads):
                if next_fn is not None and grad is not None:
                    next_fn.propagate(grad._data)

        return integrated_backward


class Function(metaclass=FunctionMeta):
    def __init__(self) -> None:
        self.ctx = Context(saved_tensors=[])  # placeholder

    def __call__(self, *args, **kwargs) -> Tensor:
        raise RuntimeError("Impossible exception.")  # overridden by FunctionMeta

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor | tuple[Tensor | None, ...]:
        raise NotImplementedError
