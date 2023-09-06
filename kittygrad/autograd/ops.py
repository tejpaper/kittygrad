from .engine import FnBackward
from ..constants import Size

import numpy as np


class NegBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(-self._grad)


class ExpBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= self._ctx.out._data
        self._next_functions[0].propagate(self._grad)


class LogBackward(FnBackward):
    def _propagate(self) -> None:
        # TODO: ref
        # log(neg) == nan, but its gradient exists like in torch
        self._grad /= self._ctx.saved_tensors[0]._data
        self._next_functions[0].propagate(self._grad)


class AddBackward(FnBackward):
    def _propagate(self) -> None:
        for next_fn in self._next_functions:
            if next_fn is not None:
                next_fn.propagate(self._grad)


class SubBackward(FnBackward):  # NegBackward + AddBackward
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


class DivBackward(FnBackward):  # PowBackward + MulBackward
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


class SumBackward(FnBackward):
    def _separate_dims(self) -> tuple[Size, Size]:
        if self._ctx.dim is None:
            dims2repeat = range(len(self._ctx.shape))
        elif isinstance(self._ctx.dim, int):
            dims2repeat = [self._ctx.dim]
        else:
            dims2repeat = self._ctx.dim

        expanded_shape = list(self._ctx.shape)
        reps = [1] * len(self._ctx.shape)

        for dim in dims2repeat:
            expanded_shape[dim] = 1
            reps[dim] = self._ctx.shape[dim]

        return expanded_shape, reps

    def _propagate(self) -> None:
        expanded_shape, reps = self._separate_dims()
        self._next_functions[0].propagate(np.tile(self._grad.reshape(expanded_shape), reps))


class MeanBackward(SumBackward):  # SumBackward + MulBackward
    def _propagate(self) -> None:
        expanded_shape, reps = super()._separate_dims()
        self._next_functions[0].propagate(np.tile(self._grad.reshape(expanded_shape), reps) / np.prod(reps))


class DotBackward(MulBackward):
    pass  # exactly the same as MulBackward due to numpy self._grad autocast


class MmBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(np.matmul(self._grad, factor_2._data.T))

        if fn_2 is not None:
            fn_2.propagate(np.matmul(factor_1._data.T, self._grad))


class MvBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            # the only difference with MmBackward is the appropriate operands shapes here
            fn_1.propagate(np.matmul(self._grad[..., np.newaxis], factor_2._data[np.newaxis, ...]))

        if fn_2 is not None:
            fn_2.propagate(np.matmul(factor_1._data.T, self._grad))

