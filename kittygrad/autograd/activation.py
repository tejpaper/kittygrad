from .engine import FnBackward


# noinspection PyProtectedMember
class SigmoidBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= self._ctx.out._data * (1 - self._ctx.out._data)
        self._next_functions[0].propagate(self._grad)


# noinspection PyProtectedMember
class TanhBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= (1 - self._ctx.out._data ** 2)
        self._next_functions[0].propagate(self._grad)


# noinspection PyProtectedMember
class ReluBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= (self._ctx.out._data > 0)
        self._next_functions[0].propagate(self._grad)
