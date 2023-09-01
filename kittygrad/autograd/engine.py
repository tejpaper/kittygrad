from __future__ import annotations

import typing

# creepy alias to avoid circular imports and provide normal variable names
import kittygrad.tensor as tsr
from ..utils import DotDict

import numpy as np

import abc
import warnings
from functools import wraps


class BackwardAccess(abc.ABC):  # ba short
    @abc.abstractmethod
    def propagate(self, prev_grad: np.ndarray) -> None:
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

    def propagate(self, grad: np.ndarray):
        if self._tensor.shape != grad.shape:
            raise RuntimeError(f"The size of tensor {self._tensor.shape} "
                               f"must match the size of its gradient {grad.shape}.")

        if self._tensor._grad is None:
            self._tensor._grad = grad
        else:
            # += will cause a bug if self._tensor._grad is grad
            self._tensor._grad = self._tensor._grad + grad


# TODO: test memory leaks, mb weak pointers are needed
class FnBackward(BackwardAccess, abc.ABC):  # fn short
    def __init__(self, ctx: dict[str, list[tsr.Tensor] | tsr.Tensor],
                 next_functions: list[FnBackward | None]) -> None:
        self._ctx = DotDict(ctx)
        self._next_functions = next_functions

        self._grad = np.zeros_like(self._ctx.out._data)
        self._lock = 0  # instead of topological sort

        self._versions = DotDict(
            out=self._ctx.out._version,
            saved_tensors=[tensor._version for tensor in self._ctx.saved_tensors]
        )

    @property  # not writable
    def next_functions(self) -> list[FnBackward | None]:
        return self._next_functions

    @abc.abstractmethod
    def _propagate(self) -> None:
        pass

    def propagate(self, prev_grad: np.ndarray) -> None:
        assert id(self._grad) != id(prev_grad)  # TODO: remove me after a bunch of tests
        self._grad += prev_grad
        self._lock -= 1

        if self._lock > 0:
            return
        elif self._lock < 0 or not self._next_functions:
            raise RuntimeError("Trying to backward through the graph a second time.")

        for tensor, old_version in zip(self._ctx.saved_tensors, self._versions.saved_tensors):
            if tensor._version != old_version:
                raise RuntimeError("One of the variables needed for gradient computation "
                                   "has been modified by an inplace operation.")

        self._propagate()

        # cut off
        self._ctx.out._grad_fn = None

        # hook
        if self._ctx.out.retains_grad:
            if self._ctx.out._version == self._versions.out:
                self._ctx.out._grad = self._grad
            else:
                warnings.warn("An attempt to assign a gradient to a tensor with retains_grad=True "
                              "and modified by inplace operation was noticed.")


def backward_graph(node: typing.Type[FnBackward]) -> typing.Callable:
    def backward_graph_decor(function: typing.Callable) -> typing.Callable:

        @wraps(function)
        def wrapper(*inputs: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
            out, *saved_tensors = function(*inputs)

            if not out.requires_grad:
                return out,

            next_functions = []

            for tensor in inputs:
                if (tensor_grad_fn := tensor.grad_fn) is not None:
                    tensor.grad_fn._lock += 1
                elif tensor.requires_grad and tensor.is_leaf:
                    tensor_grad_fn = AccumulateGrad(tensor)
                else:
                    continue

                next_functions.append(tensor_grad_fn)

            out._is_leaf = False
            out._grad_fn = node(
                ctx=dict(saved_tensors=saved_tensors, out=out),
                next_functions=next_functions,
            )

            return out,

        return wrapper
    return backward_graph_decor


def check_locks(head: FnBackward) -> bool:
    visited = set()
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
