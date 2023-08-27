from __future__ import annotations
import kittygrad.tensor as tsr  # creepy alias to avoid circular imports and provide normal variable names

import abc
import typing
import numpy as np


class AccumulateGrad:
    def __init__(self, tensor: tsr.Tensor) -> None:
        self.tensor = tensor

    def accumulate(self, grad: np.ndarray):
        if self.tensor.shape != grad.shape:
            raise RuntimeError(f'The size of tensor {self.tensor.shape} '
                               f'must match the size of its gradient {grad.shape}')

        if self.tensor.grad is None:
            self.tensor.grad = grad
        else:
            self.tensor.grad += grad


class BackwardAccess(abc.ABC):
    def __init__(self,
                 source_ctx: list[tsr.Tensor],
                 outcome_ctx: list[tsr.Tensor],  # TODO: test memory leaks, mb weak pointers are needed
                 next_fn: list[tuple[typing.Optional[BackwardAccess | AccumulateGrad], int]]
                 ) -> None:
        self.source_ctx = source_ctx  # for gradient itself
        self.outcome_ctx = outcome_ctx
        self.next_fn = next_fn

        self.versions = [tensor._version for tensor in source_ctx]  # noqa: friend

    @abc.abstractmethod
    def propagate(self):
        pass  # TODO: break weak pointer connection with tensor.grad_fn = None


class AddBackward(BackwardAccess):
    def propagate(self):
        pass  # TODO


class MulBackward(BackwardAccess):
    def propagate(self):
        pass
