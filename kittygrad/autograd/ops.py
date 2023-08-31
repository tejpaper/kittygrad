from .engine import FnBackward


class AddBackward(FnBackward):
    def _propagate(self) -> None:
        for next_fn in self._next_functions:
            if next_fn is not None:
                next_fn.propagate(self._grad)


# noinspection PyProtectedMember
class MulBackward(FnBackward):
    def _propagate(self) -> None:
        factor_1, factor_2 = self._ctx.saved_tensors
        fn_1, fn_2 = self._next_functions

        if fn_1 is not None:
            grad_1 = factor_2._data * self._grad
            fn_1.propagate(grad_1)

        if fn_2 is not None:
            grad_2 = factor_1._data * self._grad
            fn_2.propagate(grad_2)
