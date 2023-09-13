from .engine import FnBackward


class SigmoidBackward(FnBackward):
    def _propagate(self) -> None:
        # np.multiply(
        #     x1=self._grad,
        #     x2=self._ctx.out._data * (1 - self._ctx.out._data),
        #     out=self._grad,
        #     ... # TODO: continue
        # )

        self._grad *= self._ctx.out._data * (1 - self._ctx.out._data)
        self._next_functions[0].propagate(self._grad)


class TanhBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= (1 - self._ctx.out._data ** 2)
        self._next_functions[0].propagate(self._grad)


class ReluBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= (self._ctx.out._data > 0)
        self._next_functions[0].propagate(self._grad)
