from __future__ import annotations

import abc
import typing
import warnings
from collections.abc import Iterable
from contextlib import nullcontext
from functools import wraps

from inflection import underscore

from kittygrad.autograd.engine import FnBackward, BackwardGraph
from kittygrad.func.handler import normalize_args
from kittygrad.tensor.tensor import tensor
from kittygrad.utils.classes import DotDict


class no_grad:
    """
    Class enabling the temporary suspension of gradient computation within a specified scope
    or when decorating functions.
    """

    def __new__(cls, orig_func: typing.Callable | None = None) -> typing.Callable | no_grad | nullcontext:
        """
        Context manager to temporarily disable gradient computation.

        Parameters
        ----------
        orig_func : Callable or None, optional
            The original function to be decorated. If provided, the function will be called
            within the `no_grad` context, and its output tensor will have requires_grad set to False.

        Returns
        -------
        Callable or no_grad or nullcontext
            If `orig_func` is provided, returns the decorated function.
            If `orig_func` is not provided, returns either a new instance of `no_grad`
            or a null context in case of nesting.

        Examples
        --------
        1. Using `no_grad` as a context manager:
        ```python
        import kittygrad as kitty

        with kitty.no_grad():
            # Code within this block will have gradient computation disabled.
            result = some_function()
        ```

        2. Decorating a function with `no_grad`:
        ```python
        import kittygrad as kitty

        @kitty.no_grad
        def my_function():
            # Code within this function will have gradient computation disabled.
            return some_tensor_operation()
        ```

        3. Using `no_grad` as a null context when already in a `no_grad` block:
        ```python
        import kittygrad as kitty

        with kitty.no_grad():
            # Code within this block will have gradient computation disabled.
            with kitty.no_grad():
                # The second no_grad has no additional effect.
                result = another_function()
        ```
        """
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

        def integrated_forward(ctx: DotDict, self: typing.Self[Function], *args, **kwargs) -> Tensor:
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

            if not isinstance(prev_grads, Iterable):
                prev_grads = (prev_grads,)

            for next_fn, grad in zip(self._next_functions, prev_grads):
                if next_fn is not None and grad is not None:
                    next_fn.propagate(grad._data)

        return integrated_backward


class Function(metaclass=FunctionMeta):
    """
    Base class for defining custom differentiable operations.

    Subclasses must implement the `forward` and `backward` methods representing
    the forward pass computation and the backward pass gradient computation, respectively.

    Attributes
    ----------
    ctx : DotDict
        A context dictionary for saving intermediate calculations in it for future backward pass.
    """

    def __init__(self) -> None:
        self.ctx = DotDict(saved_tensors=[])  # placeholder

    def __call__(self, *args, **kwargs) -> Tensor:
        raise RuntimeError("Impossible exception.")  # overridden by FunctionMeta

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Abstract method. Perform the forward pass computation.

        Parameters
        ----------
        *args, **kwargs
            Arguments to be used in the forward pass.

        Returns
        -------
        Tensor
            Result of the forward pass computation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor | tuple[Tensor | None, ...]:
        """
        Abstract method. Perform the backward pass gradient computation.

        Parameters
        ----------
        grad_output : Tensor
            Gradient expected from the following operation.

        Returns
        -------
        Tensor or tuple[Tensor or None, ...]
            Gradient of the input tensors. The length of the tuple must match the number of input tensors.
        """
        raise NotImplementedError
