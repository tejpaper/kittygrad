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
    REL_TOL = 1e-5
    ABS_TOL = 1e-8

    def __init__(self) -> None:
        # statistics
        self.exact = 0
        self.approximate = 0
        self.max_ratio = 0
        self.ratios = list()

    def __call__(self, kitty_tensor: kitty.Tensor, torch_tensor: torch.Tensor) -> bool:
        for attr in ('requires_grad', 'is_leaf', 'retains_grad'):
            attr_value_1 = getattr(kitty_tensor, attr)
            attr_value_2 = getattr(torch_tensor, attr)

            if attr_value_1 != attr_value_2:
                print(f'\n{attr} attribute mismatch: {attr_value_1} != {attr_value_2}.')
                return False

        if self.TYPES_MAPPING[str(kitty_tensor.dtype)] != torch_tensor.dtype:
            print(f'\nTypes mismatch: {kitty_tensor.dtype} is incomparable with {torch_tensor.dtype}.')
            return False

        kitty_array = kitty_tensor._data
        torch_array = torch_tensor.detach().numpy()

        exact_match = np.array_equal(kitty_array, torch_array)

        if exact_match:
            self.exact += 1
            self.approximate += 1
            return True

        if kitty_array.shape != torch_array.shape:
            print(f'\nShapes mismatch: {kitty_array.shape} != {torch_array.shape}.')
            return False

        diff = np.abs(kitty_array - torch_array)
        torch_array_abs = np.abs(torch_array)
        approximate_match = (diff <= self.ABS_TOL + self.REL_TOL * torch_array_abs).all()

        # ratio must be less than 1 (critical value), it indicates how different the arrays are
        ratios = (diff - self.ABS_TOL) / torch_array_abs / self.REL_TOL
        self.max_ratio = max(self.max_ratio, np.max(ratios))
        self.ratios.append(np.mean(ratios))

        self.approximate += approximate_match
        return approximate_match

    def __str__(self) -> str:
        return (f'Exact: {self.exact} | '
                f'approximate: {self.approximate} | '
                f'maximum ratio: {self.max_ratio:.4f} | '
                f'mean ratio: {(np.mean(self.ratios) if self.ratios else 0):.4f}')


@pytest.fixture
def compare():
    cmp = Comparison()
    yield cmp
    print('\n', cmp, sep='', end='')


def init_tensors(shapes: list[kitty.Size], dtypes: list[np.dtype] = None, squeeze_dims: kitty.Size = None):
    if dtypes is None:
        dtypes = [np.float32] * len(shapes)

    if squeeze_dims is None:
        squeeze_dims = tuple()

    np.random.seed(42)
    data = [np.random.randn(*shape).astype(dtype).squeeze(squeeze_dims) for shape, dtype in zip(shapes, dtypes)]
    return map(lambda lib: map(lambda d: lib.tensor(d, requires_grad=True), data), (kitty, torch))
