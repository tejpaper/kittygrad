from kittygrad.autograd.engine import FnBackward


class SigmoidBackward(FnBackward):
    def _propagate(self) -> None:
        self._inplace_modification_check()
        self._grad *= self._ctx.out._data * (1 - self._ctx.out._data)
        self._next_functions[0].propagate(self._grad)


class TanhBackward(FnBackward):
    def _propagate(self) -> None:
        self._inplace_modification_check()
        self._grad *= (1 - self._ctx.out._data ** 2)
        self._next_functions[0].propagate(self._grad)


class ReluBackward(FnBackward):
    def _propagate(self) -> None:
        self._inplace_modification_check()
        self._grad *= (self._ctx.out._data > 0)
        self._next_functions[0].propagate(self._grad)
