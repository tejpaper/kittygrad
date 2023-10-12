from __future__ import annotations

import abc
import warnings
from functools import update_wrapper, wraps

import kittygrad.tensor as tsr
from ..utils import *


class BackwardAccess(abc.ABC):  # ba short
    @abc.abstractmethod
    def propagate(self, prev_grad: np.ndarray | np.generic) -> None:
        raise NotImplementedError

    def __format__(self, format_spec: str) -> str:
        return f'<{type(self).__name__}>'.__format__(format_spec)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} object at 0x{id(self):02x}>'

    def __str__(self) -> str:
        return repr(self)


class AccumulateGrad(BackwardAccess):  # ag short
    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    def propagate(self, grad: np.ndarray | np.generic) -> None:
        # TODO: remove me after a bunch of tests
        if self._tensor.shape != grad.shape:
            raise RuntimeError(f"The size of tensor {self._tensor.shape} "
                               f"must match the size of its gradient {grad.shape}.")

        if self._tensor._grad is None:
            self._tensor._grad = np.zeros_like(self._tensor._data)

        # += will cause a bug if self._tensor._grad is grad
        np.add(self._tensor._grad, grad, out=self._tensor._grad, **NP_OPS_CONFIG)


class FnBackward(BackwardAccess, abc.ABC):  # fn short
    def __init__(self,
                 ctx: DotDict[str, list[typing.Any] | Tensor],
                 next_functions: list[FnBackward | None]) -> None:
        self._ctx = ctx
        self._next_functions = next_functions

        self._grad = np.zeros_like(self._ctx.out._data)
        self._lock = 0  # instead of topological sort

        self._versions = DotDict(
            out=self._ctx.out.version,
            saved_tensors=[
                tensor.version if tensor is not None else 0
                for tensor in self._ctx.saved_tensors
            ],
        )

    @property  # not writable
    def next_functions(self) -> tuple[FnBackward | None, ...]:
        return tuple(self._next_functions)

    @abc.abstractmethod
    def _propagate(self) -> None:
        raise NotImplementedError  # self._grad can be changed here as there is a hook before it

    def _inplace_modification_check(self) -> None:
        if self._ctx.out.version != self._versions.out:
            inplace_modification_error()

    def propagate(self, prev_grad: np.ndarray | np.generic) -> None:
        assert id(self._grad) != id(prev_grad)  # TODO: remove me after a bunch of tests
        np.add(self._grad, prev_grad, out=self._grad, **NP_OPS_CONFIG)
        self._lock -= 1

        if self._lock > 0:
            return
        elif self._lock < 0 or all(next_fn is None for next_fn in self._next_functions):
            redundant_backward_error()

        for tensor, old_version in zip(self._ctx.saved_tensors, self._versions.saved_tensors):
            if tensor is not None and tensor.version != old_version:
                inplace_modification_error()

        # retain_grad hook
        if self._ctx.out.retains_grad:
            if self._ctx.out.version == self._versions.out:
                self._ctx.out._grad = self._grad.copy()  # no ref to avoid bugs
            else:
                warnings.warn("An attempt to assign a gradient to a tensor with retains_grad=True "
                              "and modified by inplace operation was noticed.")

        self._propagate()

        # cut off
        self._ctx.out._grad_fn = None


class BackwardGraph:
    pre_builder_hooks = DotDict()
    post_builder_hooks = DotDict()
    disabled = False

    def __init__(self, function: typing.Callable, node: typing.Type[FnBackward]) -> None:
        self.function = function
        self.node = node
        update_wrapper(self, function)

    def __call__(self, *args) -> Tensor:
        ctx = DotDict(saved_tensors=[])
        out = self.function(*args, ctx)

        if not out.requires_grad:
            return out

        out._is_leaf = False

        ctx.out = out
        next_functions = []

        for arg in args:
            if not isinstance(arg, tsr.Tensor):
                continue

            if (tensor_grad_fn := arg.grad_fn) is not None:
                arg.grad_fn._lock += 1
            elif arg.requires_grad and arg.is_leaf:
                tensor_grad_fn = AccumulateGrad(arg)

            next_functions.append(tensor_grad_fn)

        out._grad_fn = self.node(ctx=ctx, next_functions=next_functions)
        return out

    @staticmethod
    def mount(node: typing.Type[FnBackward]) -> typing.Callable:
        def backward_graph(function: typing.Callable) -> typing.Callable:
            @wraps(function)
            def apply_hooks(*args) -> Tensor:

                if BackwardGraph.disabled:
                    dummy_ctx = DotDict(saved_tensors=[])
                    return function(*args, dummy_ctx)

                wrapped_func = function

                for hook in BackwardGraph.pre_builder_hooks.values():
                    wrapped_func = hook(wrapped_func)

                wrapped_func = BackwardGraph(wrapped_func, node)

                for hook in BackwardGraph.post_builder_hooks.values():
                    wrapped_func = hook(wrapped_func)

                return wrapped_func(*args)
            return apply_hooks
        return backward_graph

    @staticmethod
    def disable(function: typing.Callable) -> typing.Callable:
        @wraps(function)
        def decorated(*args, **kwargs) -> typing.Any:
            BackwardGraph.disabled = True
            output = function(*args, **kwargs)
            BackwardGraph.disabled = False

            return output
        return decorated


def check_locks(head: FnBackward) -> None:
    visited = {None}
    queue = {head}

    while queue:
        fn = queue.pop()
        visited.add(fn)

        if fn._lock > 0:
            warnings.warn("Backpropagation not completed. The computational graph "
                          "has at least one more output for the .backward() call.")
            return

        for ba in fn._next_functions:
            if not isinstance(ba, AccumulateGrad) and ba not in visited:
                queue.add(ba)
