from .engine import FnBackward
from ..utils import inv_permutation


import numpy as np


class TransposeBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(np.swapaxes(self._grad, self._ctx.dim0, self._ctx.dim1))


class PermuteBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(np.transpose(self._grad, inv_permutation(self._ctx.dims)))
