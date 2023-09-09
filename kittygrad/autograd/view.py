from .engine import FnBackward
from ..utils import inv_permutation


import numpy as np


class TransposeBackward(FnBackward):  # PermuteBackward
    def _propagate(self) -> None:
        self._next_functions[0].propagate(np.swapaxes(self._grad, self._ctx.dim0, self._ctx.dim1))


class PermuteBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(np.transpose(self._grad, inv_permutation(self._ctx.dims)))


class SqueezeBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(self._grad.reshape(self._ctx.shape))


class UnsqueezeBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(self._grad.squeeze(self._ctx.dim))


class ExpandBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(
            self._grad.sum(self._ctx.expanded_dims, keepdims=True).squeeze(self._ctx.leading_dims)
        )
