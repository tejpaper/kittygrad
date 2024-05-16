import kittygrad.core as core
from kittygrad.autograd.engine import FnBackward
from kittygrad.func.utils import separate_dims


class ToCopyBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(self._grad.astype(self._ctx.prev_dtype))


class CloneBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(self._grad)


class NegBackward(FnBackward):
    def _propagate(self) -> None:
        self._next_functions[0].propagate(-self._grad)


class AbsBackward(FnBackward):
    def _propagate(self) -> None:
        self._grad *= core.strict.sign(self._ctx.saved_tensors[0]._data)
        self._next_functions[0].propagate(self._grad)


class ExpBackward(FnBackward):
    def _propagate(self) -> None:
        self._inplace_modification_check()
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


class IMulBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_arrays
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(factor_2 * self._grad)

        if fn_2 is not None:
            fn_2.propagate(factor_1 * self._grad)


class DivBackward(FnBackward):
    def _propagate(self) -> None:
        dividend_fn, divisor_fn = self._next_functions

        self._grad *= self._ctx.other_inv

        if dividend_fn is not None:
            dividend_fn.propagate(self._grad)

        if divisor_fn is not None:
            self._inplace_modification_check()
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
        self._inplace_modification_check()

        base, exponent = self._ctx.saved_tensors
        base_fn, exponent_fn = self._next_functions

        self._grad *= self._ctx.out._data

        if base_fn is not None:
            base_fn.propagate(exponent._data / base._data * self._grad)

        if exponent_fn is not None:
            exponent_fn.propagate(core.strict.log(base._data) * self._grad)


class IPowBackward(FnBackward):
    def _propagate(self) -> None:
        base, exponent = self._ctx.saved_arrays
        base_fn, exponent_fn = self._next_functions

        self._grad *= self._ctx.out_array

        if base_fn is not None:
            base_fn.propagate(exponent / base * self._grad)

        if exponent_fn is not None:
            exponent_fn.propagate(core.strict.log(base) * self._grad)


class SumBackward(FnBackward):
    def _propagate(self) -> None:
        expanded_shape, reps = separate_dims(self._ctx.shape, self._ctx.dim)
        self._next_functions[0].propagate(core.np.tile(self._grad.reshape(expanded_shape), reps))


class MeanBackward(FnBackward):
    def _propagate(self) -> None:
        expanded_shape, reps = separate_dims(self._ctx.shape, self._ctx.dim)
        self._next_functions[0].propagate(
            core.np.tile(self._grad.reshape(expanded_shape), reps) / core.np.prod(reps, dtype=self._grad.dtype))


class VarBackward(FnBackward):
    def _var_std_propagate(self) -> None:
        self._grad = core.np.tile(self._grad.reshape(self._ctx.expanded_shape), self._ctx.reps)
        self._grad *= self._ctx.residuals
        self._grad += core.np.sum(self._grad, axis=self._ctx.dim, keepdims=True) / self._ctx.n
        self._next_functions[0].propagate(self._grad)

    def _propagate(self) -> None:
        self._grad *= 2 / (self._ctx.n - self._ctx.correction)
        self._var_std_propagate()


class StdBackward(VarBackward):
    def _propagate(self) -> None:
        self._inplace_modification_check()

        self._grad /= (self._ctx.n - self._ctx.correction)
        self._grad /= self._ctx.out._data
        self._var_std_propagate()


class MmBackward(FnBackward):
    def _propagate(self) -> None:
        matrix_1, matrix_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            fn_1.propagate(core.strict.matmul(self._grad, core.np.swapaxes(matrix_2._data, -2, -1)))

        if fn_2 is not None:
            fn_2.propagate(core.strict.matmul(core.np.swapaxes(matrix_1._data, -2, -1), self._grad))


class DotBackward(MulBackward):
    pass  # exactly the same as MulBackward due to numpy self._grad autocast


class MvBackward(FnBackward):
    def _propagate(self) -> None:
        matrix, vector = self._ctx.saved_tensors
        matrix_fn, vector_fn = self._next_functions

        if matrix_fn is not None:
            matrix_fn.propagate(core.np.outer(self._grad, vector._data))

        if vector_fn is not None:
            vector_fn.propagate(core.strict.matmul(matrix._data.T, self._grad))


class BmmBackward(MmBackward):
    pass  # exactly the same as MmBackward, since np.swapaxes is used instead of .T
