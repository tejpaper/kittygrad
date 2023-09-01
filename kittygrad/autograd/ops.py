from .engine import FnBackward


class AddBackward(FnBackward):
    def _propagate(self) -> None:
        for next_fn in self._next_functions:
            next_fn.propagate(self._grad)


class MulBackward(FnBackward):
    def _propagate(self) -> None:
        for factor, next_fn in zip(self._ctx.saved_tensors, self._next_functions):
            next_fn.propagate(factor._data * self._grad)
