from .engine import FnBackward
from ..utils import *


class TransposeBackward(FnBackward):
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


class IndexBackward(FnBackward):
    def _propagate(self) -> None:
        extended_grad = np.zeros(self._ctx.shape, dtype=self._grad.dtype)
        np.add.at(extended_grad, self._ctx.key, self._grad)
        self._next_functions[0].propagate(extended_grad)


class IndexPutBackward(FnBackward):
    def _propagate(self) -> None:
        base_fn, value_fn = self._next_functions

        if value_fn is not None:
            value_fn.propagate(self._grad[self._ctx.key])

        if base_fn is not None:
            self._grad[self._ctx.key] = 0
            base_fn.propagate(self._grad)
