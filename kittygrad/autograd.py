from __future__ import annotations
import kittygrad.tensor as tsr  # creepy alias to avoid circular imports and provide normal variable names

import abc
import typing
import warnings
import numpy as np


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


# noinspection PyProtectedMember
class FnBackward(BackwardAccess, abc.ABC):  # fn short
    def __init__(self,
                 source_ctx: list[tsr.Tensor],
                 outcome_ctx: tsr.Tensor,  # TODO: test memory leaks, mb weak pointers are needed
                 next_functions: list[typing.Optional[FnBackward]]
                 ) -> None:
        self._source_ctx = source_ctx  # for gradient calculation
        self._outcome_ctx = outcome_ctx  # for cut off
        self._next_functions = next_functions

        self._grad = np.zeros_like(outcome_ctx._data)
        self._lock = 0  # instead of topological sort

        # TODO: version control
        self._versions = [tensor._version for tensor in source_ctx]

    @property  # not writable
    def next_functions(self) -> list[typing.Optional[FnBackward]]:
        return self._next_functions

    @abc.abstractmethod
    def _propagate(self) -> None:
        pass

    def propagate(self, prev_grad: np.ndarray) -> None:
        self._grad += prev_grad
        self._lock -= 1

        if self._lock < 0:  # TODO: remove
            warnings.warn(f"Negative lock value on {repr(self)}")

        if self._lock <= 0:
            self._propagate()

            # cut off
            self._outcome_ctx._grad_fn = None

            # hook
            if self._outcome_ctx.retains_grad:
                self._outcome_ctx._grad = self._grad


class AddBackward(FnBackward):
    def _propagate(self) -> None:
        for next_fn in self._next_functions:
            if next_fn is not None:
                next_fn.propagate(self._grad)


# noinspection PyProtectedMember
class MulBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._source_ctx
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            grad_1 = factor_2._data * self._grad
            fn_1.propagate(grad_1)

        if fn_2 is not None:
            grad_2 = factor_1._data * self._grad
            fn_2.propagate(grad_2)
