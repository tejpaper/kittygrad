from __future__ import annotations

import typing

# creepy alias to avoid circular imports and provide normal variable names
import kittygrad.tensor as tsr
from ..utils import DotDict, inplace_modification_error

import numpy as np

import abc
import warnings
from functools import wraps


class BackwardAccess(abc.ABC):  # ba short
    @abc.abstractmethod
    def propagate(self, prev_grad: np.ndarray | np.generic) -> None:
        pass

    def __format__(self, format_spec):
        return f'<{type(self).__name__}>'.__format__(format_spec)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} object at 0x{id(self):02x}>'

    def __str__(self) -> str:
        return repr(self)


class AccumulateGrad(BackwardAccess):  # ag short
    def __init__(self, tensor: tsr.Tensor) -> None:
        self._tensor = tensor

    def propagate(self, grad: np.ndarray | np.generic) -> None:
        # TODO: remove us after a bunch of tests
        assert self._tensor.dtype == grad.dtype
        if self._tensor.shape != grad.shape:
            raise RuntimeError(f"The size of tensor {self._tensor.shape} "
                               f"must match the size of its gradient {grad.shape}.")

        if self._tensor._grad is None:
            self._tensor._grad = np.zeros_like(self._tensor._data)

        # += will cause a bug if self._tensor._grad is grad
        np.add(self._tensor._grad, grad, out=self._tensor._grad)


# TODO: test memory leaks, mb weak pointers are needed
class FnBackward(BackwardAccess, abc.ABC):  # fn short
    def __init__(self,
                 ctx: DotDict[str, list[typing.Any] | tsr.Tensor],
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
    def next_functions(self) -> list[FnBackward | None]:
        return self._next_functions

    @abc.abstractmethod
    def _propagate(self) -> None:
        pass  # self._grad can be changed here as there is a hook before it

    def propagate(self, prev_grad: np.ndarray | np.generic) -> None:
        assert id(self._grad) != id(prev_grad)  # TODO: remove me after a bunch of tests
        self._grad += prev_grad
        self._lock -= 1

        if self._lock > 0:
            return
        elif self._lock < 0 or all(next_fn is None for next_fn in self._next_functions):
            raise RuntimeError("Trying to backward through the graph a second time.")

        for tensor, old_version in zip(self._ctx.saved_tensors, self._versions.saved_tensors):
            if tensor is not None and tensor.version != old_version:
                inplace_modification_error()

        # hook
        if self._ctx.out.retains_grad:
            if self._ctx.out.version == self._versions.out:
                self._ctx.out._grad = np.copy(self._grad)  # no ref to avoid bugs
            else:
                warnings.warn("An attempt to assign a gradient to a tensor with retains_grad=True "
                              "and modified by inplace operation was noticed.")

        self._propagate()

        # cut off
        self._ctx.out._grad_fn = None


def backward_graph(node: typing.Type[FnBackward]) -> typing.Callable:
    def backward_graph_decor(function: typing.Callable) -> typing.Callable:

        @wraps(function)
        def wrapper(*args) -> tsr.Tensor:
            ctx = DotDict(saved_tensors=[])
            out = function(*args, ctx)

            if not out.requires_grad:
                return out

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

            out._is_leaf = False
            out._grad_fn = node(ctx=ctx, next_functions=next_functions)

            return out

        return wrapper
    return backward_graph_decor


def check_locks(head: FnBackward) -> bool:
    visited = {None}
    queue = {head}

    while queue:
        fn = queue.pop()
        visited.add(fn)

        if fn._lock > 0:
            return True

        for ba in fn._next_functions:
            if not isinstance(ba, AccumulateGrad) and ba not in visited:
                queue.add(ba)

    return False
