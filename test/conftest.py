import numpy as np
import pytest
import torch

import kittygrad as kitty


class Comparison:
    TYPES_MAPPING = {
        kitty.float16.__name__: torch.float16,
        kitty.half.__name__: torch.half,
        kitty.float32.__name__: torch.float32,
        kitty.float.__name__: torch.float,
        kitty.float64.__name__: torch.float64,
        kitty.double.__name__: torch.double,
    }

    def __init__(self, relative_tol: float = 1e-5, absolute_tol: float = 1e-8) -> None:
        self.exact = 0
        self.approximate = 0
        self.max_rel_diff = 0

        self.rel_tol = relative_tol
        self.abs_tol = absolute_tol

    def __call__(self, kitty_tensor: kitty.Tensor, torch_tensor: torch.Tensor) -> bool:
        for attr in ('requires_grad', 'is_leaf', 'retains_grad'):
            attr_value_1 = getattr(kitty_tensor, attr)
            attr_value_2 = getattr(torch_tensor, attr)

            if attr_value_1 != attr_value_2:
                print(f'{attr} attribute mismatch: {attr_value_1} != {attr_value_2}.')
                return False

        if self.TYPES_MAPPING[str(kitty_tensor.dtype)] != torch_tensor.dtype:
            print(f'Types mismatch: {kitty_tensor.dtype} is incomparable with {torch_tensor.dtype}.')
            return False

        kitty_array = kitty_tensor._data
        torch_array = torch_tensor.detach().numpy()

        exact_match = np.array_equal(kitty_array, torch_array)

        if exact_match:
            self.exact += 1
            self.approximate += 1
            return True

        if kitty_array.shape != torch_array.shape:
            print(f'Shapes mismatch: {kitty_array.shape} != {torch_array.shape}.')
            return False

        approximate_match = np.allclose(kitty_array, torch_array, self.rel_tol, self.abs_tol)

        self.approximate += approximate_match
        self.max_rel_diff = max(self.max_rel_diff,
                                np.max(self.rel_tol - np.abs(kitty_array - torch_array) / np.abs(torch_array)))

        return approximate_match

    def __str__(self) -> str:
        return (f'Exact: {self.exact} | '
                f'approximate: {self.approximate} | '
                f'maximum relative difference: {self.max_rel_diff}')


@pytest.fixture
def compare():
    cmp = Comparison()
    yield cmp
    print(cmp)


def init_tensors(*shapes: kitty.Size, squeeze_dims: kitty.Size = None):
    if squeeze_dims is None:
        squeeze_dims = tuple()

    np.random.seed(42)
    data = [np.random.randn(*shape).astype(np.float32).squeeze(squeeze_dims) for shape in shapes]
    return map(lambda lib: map(lambda d: lib.tensor(d, requires_grad=True), data), (kitty, torch))
