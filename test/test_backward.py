import pytest
import torch
import numpy as np
import kittygrad as kitty


class Comparison:
    def __init__(self, relative_tol: float = 1e-5, absolute_tol: float = 1e-8) -> None:
        self.exact = 0
        self.approximate = 0
        self.max_rel_diff = 0

        self.rel_tol = relative_tol
        self.abs_tol = absolute_tol

    def __call__(self, kitty_tensor: kitty.Tensor, torch_tensor: torch.Tensor, is_grad: bool = False) -> bool:
        for attr in ('requires_grad', 'is_leaf', 'retains_grad'):
            attr_value_1 = getattr(kitty_tensor, attr)
            attr_value_2 = getattr(torch_tensor, attr)

            if attr_value_1 != attr_value_2:
                print(f'{attr} attribute mismatch: {attr_value_1} != {attr_value_2}.')
                return False

        if not is_grad and kitty_tensor.version != torch_tensor._version:
            print(f'Versions mismatch: {kitty_tensor.version} != {torch_tensor._version}.')
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

        self.max_rel_diff = max(self.max_rel_diff,
                                np.max(self.rel_tol - np.abs(kitty_array - torch_array) / np.abs(torch_array)))
        self.approximate += approximate_match

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


@pytest.fixture(
    params=(
            ((13, 1, 3), (2, 3)),
    )
)
def init_tensors(request):
    np.random.seed(42)
    data = [np.random.randn(*shape).astype(np.float32) for shape in request.param]
    yield map(lambda lib: map(lambda d: lib.tensor(d, requires_grad=True), data), (kitty, torch))


def test_ops(init_tensors, compare):
    print()
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors

    def zero_grad():
        for tensor in (kitty_a, kitty_b, torch_a, torch_b):
            tensor.grad = None

    # __pos__, __neg__, ones_like
    kitty_c = +(-kitty_a)
    torch_c = +(-torch_a)
    assert compare(kitty_c, torch_c)
    kitty_c.backward(kitty.ones_like(kitty_c))
    torch_c.backward(torch.ones_like(torch_c))
    assert compare(kitty_a.grad, torch_a.grad, is_grad=True)

    zero_grad()

    # exp, sum
    kitty_c = kitty_a.exp()
    torch_c = torch_a.exp()
    assert compare(kitty_c, torch_c)
    kitty_c.sum().backward()
    torch_c.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad, is_grad=True)

    zero_grad()

    # __pow__, log, mean
    kitty_c = (kitty_a ** 2).log()
    torch_c = (torch_a ** 2).log()
    assert compare(kitty_c, torch_c)
    kitty_c.mean().backward()
    torch_c.mean().backward()
    assert compare(kitty_a.grad, torch_a.grad, is_grad=True)

    zero_grad()

    # __add__, __mul__, __rsub__, mean, retain_grad
    kitty_c = kitty_a + kitty_b
    kitty_d = kitty_c * kitty_b
    kitty_e = (2 - kitty_d).mean()

    torch_c = torch_a + torch_b
    torch_d = torch_c * torch_b
    torch_e = (2 - torch_d).mean()

    for kitty_t, torch_t in zip((kitty_c, kitty_d, kitty_e), (torch_c, torch_d, torch_e)):
        kitty_t.retain_grad()
        torch_t.retain_grad()
        assert compare(kitty_t, torch_t)

    kitty_e.backward()
    torch_e.backward()

    for kitty_t, torch_t in zip((kitty_a, kitty_b, kitty_c, kitty_d, kitty_e),
                                (torch_a, torch_b, torch_c, torch_d, torch_e)):
        assert compare(kitty_t.grad, torch_t.grad, is_grad=True)

    zero_grad()

    # __truediv__, __rtruediv__, __add__, sum
    kitty_c = kitty_a / 10
    kitty_d = 1 / kitty_b
    kitty_e = kitty_c + kitty_d

    torch_c = torch_a / 10
    torch_d = 1 / torch_b
    torch_e = torch_c + torch_d

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)
    assert compare(kitty_e, torch_e)

    kitty_e.sum().backward()
    torch_e.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad, is_grad=True)
    assert compare(kitty_b.grad, torch_b.grad, is_grad=True)

    zero_grad()

    # __rpow__, __radd__, __rmul__, __sub__, mean
    kitty_c = 21 + 0.5 ** (kitty_b - 0.5 * kitty_b)
    torch_c = 21 + 0.5 ** (torch_b - 0.5 * torch_b)
    assert compare(kitty_c, torch_c)

    kitty_c.mean().backward()
    torch_c.mean().backward()
    assert compare(kitty_b.grad, torch_b.grad, is_grad=True)

    zero_grad()

    # __matmul__, squeeze, mT, ones_like
    kitty_c = kitty_a.squeeze(1) @ kitty_b.mT
    torch_c = torch_a.squeeze(1) @ torch_b.mT
    assert compare(kitty_c, torch_c)

    kitty_c.backward(kitty.ones_like(kitty_c) * 0.3)
    torch_c.backward(torch.ones_like(torch_c) * 0.3)
    assert compare(kitty_a.grad, torch_a.grad, is_grad=True)
    assert compare(kitty_b.grad, torch_b.grad, is_grad=True)


def test_ops_exceptions():
    pass


def test_activation():
    pass


def test_activation_exceptions():
    pass


def test_inplace():
    pass


def test_inplace_exceptions():
    pass


def test_view():
    pass


def test_view_exceptions():
    pass
