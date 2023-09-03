from .engine import FnBackward

import numpy as np


class NegBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(-self._grad)


class ExpBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= self._ctx.out._data
        self._next_functions[0].propagate(self._grad)


class AddBackward(FnBackward):
    def _propagate(self) -> None:
        for next_fn in self._next_functions:
            if next_fn is not None:
                next_fn.propagate(self._grad)


class SubBackward(FnBackward):
    def _propagate(self) -> None:
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(self._grad)

        if fn_2 is not None:
            fn_2.propagate(-self._grad)


class MulBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(factor_2._data * self._grad)

        if fn_2 is not None:
            fn_2.propagate(factor_1._data * self._grad)


class DivBackward(FnBackward):
    def _propagate(self) -> None:
        fn_1, fn_2 = self._next_functions

        self._grad *= self._ctx.saved_arrays[0]

        if fn_1 is not None:
            fn_1.propagate(self._grad)

        if fn_2 is not None:
            fn_2.propagate(-self._ctx.out._data * self._grad)


class PowBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        self._grad *= self._ctx.out._data

        if fn_1 is not None:
            fn_1.propagate(factor_2._data / factor_1._data * self._grad)

        if fn_2 is not None:
            fn_2.propagate(np.log(factor_1._data) * self._grad)

