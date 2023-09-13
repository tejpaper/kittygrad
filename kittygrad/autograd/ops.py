from .engine import FnBackward
from ..utils import *


class ToCopyBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(self._grad.astype(self._ctx.prev_dtype))


class NegBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(-self._grad)


class ExpBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= self._ctx.out._data
        self._next_functions[0].propagate(self._grad)


class LogBackward(FnBackward):
    def _propagate(self) -> None:
        # TODO: ref [1]
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


class IMulBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_arrays
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(factor_2 * self._grad)

        if fn_2 is not None:
            fn_2.propagate(factor_1 * self._grad)


class DivBackward(FnBackward):  # PowBackward + MulBackward
    def _propagate(self) -> None:
        dividend_fn, divisor_fn = self._next_functions

        self._grad *= self._ctx.other_inv

        if dividend_fn is not None:
            dividend_fn.propagate(self._grad)

        if divisor_fn is not None:
            if self._ctx.out.version != self._versions.out:
                inplace_modification_error()
            divisor_fn.propagate(-self._ctx.out._data * self._grad)


class IDivBackward(FnBackward):
    def _propagate(self) -> None:
        dividend_fn, divisor_fn = self._next_functions

        self._grad *= self._ctx.other_inv

        if dividend_fn is not None:
            dividend_fn.propagate(self._grad)

        if divisor_fn is not None:
            divisor_fn.propagate(-self._ctx.out_array * self._grad)


class PowBackward(FnBackward):
    def _propagate(self) -> None:
        if self._ctx.out.version != self._versions.out:
            inplace_modification_error()

        base, exponent = self._ctx.saved_tensors
        base_fn, exponent_fn = self._next_functions

        self._grad *= self._ctx.out._data

        if base_fn is not None:
            base_fn.propagate(exponent._data / base._data * self._grad)

        if exponent_fn is not None:
            exponent_fn.propagate(np.log(base._data) * self._grad)


class IPowBackward(FnBackward):
    def _propagate(self) -> None:
        base, exponent = self._ctx.saved_arrays
        base_fn, exponent_fn = self._next_functions

        self._grad *= self._ctx.out_array

        if base_fn is not None:
            base_fn.propagate(exponent / base * self._grad)

        if exponent_fn is not None:
            exponent_fn.propagate(np.log(base) * self._grad)


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
        matrix_1, matrix_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(np.matmul(self._grad, np.swapaxes(matrix_2._data, -2, -1)))

        if fn_2 is not None:
            fn_2.propagate(np.matmul(np.swapaxes(matrix_1._data, -2, -1), self._grad))


class MvBackward(FnBackward):
    def _propagate(self) -> None:
        matrix, vector = self._ctx.saved_tensors
        matrix_fn, vector_fn = self._next_functions

        if matrix_fn is not None:
            matrix_fn.propagate(np.matmul(self._grad[..., np.newaxis], vector._data[np.newaxis, ...]))

        if vector_fn is not None:
            vector_fn.propagate(np.matmul(matrix._data.T, self._grad))


class BmmBackward(MmBackward):
    pass  # exactly the same as MmBackward, since np.swapaxes is used instead of .T
