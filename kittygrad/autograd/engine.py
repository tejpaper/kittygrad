from __future__ import annotations

import typing

# creepy alias to avoid circular imports and provide normal variable names
import kittygrad.tensor as tsr
from ..utils import DotDict

import numpy as np

import abc
from functools import wraps


def backward_graph(node: typing.Type[FnBackward]) -> typing.Callable:
    def backward_graph_decor(function: typing.Callable) -> typing.Callable:

        # noinspection PyProtectedMember
        @wraps(function)
        def wrapper(*inputs: tsr.Tensor) -> tsr.Tensor:
            out, *saved_tensors = function(*inputs)

            if not out.requires_grad:
                return out

            next_functions = []

            for tensor in inputs:
                if (tensor_grad_fn := tensor.grad_fn) is not None:
                    tensor.grad_fn._lock += 1
                elif tensor.is_leaf:
                    tensor_grad_fn = AccumulateGrad(tensor)

                next_functions.append(tensor_grad_fn)

            out._is_leaf = False
            out._grad_fn = node(
                ctx=dict(saved_tensors=saved_tensors, out=out),
                next_functions=next_functions,
            )

            return out

        return wrapper
    return backward_graph_decor


class BackwardAccess(abc.ABC):  # ba short
    @abc.abstractmethod
    def propagate(self, prev_grad: np.ndarray) -> None:
        pass

    def __repr__(self) -> str:  # TODO: torch-like
        return f'<{type(self).__name__}>'

    def __str__(self) -> str:
        return repr(self)


# noinspection PyProtectedMember
class AccumulateGrad(BackwardAccess):  # ag short
    def __init__(self, tensor: tsr.Tensor) -> None:
        self._tensor = tensor
        self._version = tensor._version  # TODO: version control

    def propagate(self, grad: np.ndarray):
        if self._tensor.shape != grad.shape:
            raise RuntimeError(f"The size of tensor {self._tensor.shape} "
                               f"must match the size of its gradient {grad.shape}")

        if self._tensor._grad is None:
            self._tensor._grad = grad
        else:
            # += will cause a bug if self._tensor._grad is grad
            self._tensor._grad = self._tensor._grad + grad


# TODO: test memory leaks, mb weak pointers are needed
# noinspection PyProtectedMember
class FnBackward(BackwardAccess, abc.ABC):  # fn short
    def __init__(self, ctx: dict[str, list[tsr.Tensor] | tsr.Tensor],
                 next_functions: list[FnBackward | None]) -> None:
        self._ctx = DotDict(ctx)
        self._next_functions = next_functions

        self._grad = np.zeros_like(self._ctx.out._data)
        self._lock = 0  # instead of topological sort

        # TODO: version control
        self._versions = [tensor._version for tensor in self._ctx.saved_tensors]

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

        if self._lock <= 0:
            self._propagate()

            # cut off
            self._ctx.out._grad_fn = None

            # hook
            if self._ctx.out.retains_grad:
                self._ctx.out._grad = self._grad
