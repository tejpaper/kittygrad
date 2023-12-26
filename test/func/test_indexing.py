import itertools

from kittygrad.utils.constants import ALL_DTYPES
from test.conftest import *


@pytest.mark.parametrize(
    'dtypes',
    itertools.product(ALL_DTYPES, ALL_DTYPES, ALL_DTYPES))
@pytest.mark.parametrize(
    'shapes', [
        [(4, 5, 6)],
    ])
@pytest.mark.parametrize(
    'key', [
        # basic
        np.s_[1],
        np.s_[(1,)],
        np.s_[1],
        np.s_[1],
        np.s_[:],
        np.s_[:, :],
        np.s_[:, ::2],
        np.s_[...],
        np.s_[:, 0, ...],
        np.s_[1, 2, 3],
        np.s_[(1, 2, 3)],
        np.s_[None],
        np.s_[None, 0],
        np.s_[(None, 0)],
        # advanced
        np.s_[(1, 0, 0),],
        np.s_[[1]],
        np.s_[[1, 0, 0]],
        np.s_[[1, 0, 0], [0, 0, 0]],
        np.s_[..., [0, 0, 2]],
        np.s_[[1, 1, 1], :, [0, 0, 2]],
        np.s_[[0, 0], [1, 1], [2, 2]],
    ]
)
def test_indexing_1(key, shapes, dtypes, compare):
    kitty_a, torch_a = map(next, init_tensors(shapes, dtypes))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    def test():
        assert compare(kitty_b, torch_b)

        kitty_b.sum().backward()
        torch_b.sum().backward()
        assert compare(kitty_a.grad, torch_a.grad)

        zero_grad()

    # get by index
    kitty_b = kitty_a[key]
    torch_b = torch_a[key]
    test()

    # set by index
    kitty_b = kitty_a + 0
    kitty_b[key] = 100
    torch_b = torch_a + 0
    torch_b[key] = 100
    test()


def test_indexing_2(compare):
    kitty_a, torch_a = map(next, init_tensors([(4, 5, 6)]))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    def test():
        assert compare(kitty_b, torch_b)

        kitty_b.sum().backward()
        torch_b.sum().backward()
        assert compare(kitty_a.grad, torch_a.grad)

        zero_grad()

    # numpy vs torch indexing rules
    kitty_b = kitty_a[1, ..., [0, 0, 2]]
    torch_b = torch_a[1, ..., [0, 0, 2]].T
    test()

    # boolean indexing
    kitty_b = kitty_a + 0
    kitty_b[kitty_a > 0] = 100
    torch_b = torch_a + 0
    torch_b[torch_b > 0] = 100
    test()


def test_setitem(compare):
    kitty_a, torch_a = map(next, init_tensors([(4, 5, 6)]))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    def test():
        assert compare(kitty_b, torch_b)

        kitty_b.sum().backward()
        torch_b.sum().backward()
        assert compare(kitty_a.grad, torch_a.grad)

        zero_grad()

    # gradient flow through the basic indexing operation
    kitty_b = kitty_a + 0
    kitty_b[:, :2, 3] *= kitty_a[:, :2, 3]
    torch_b = torch_a + 0
    torch_b[:, :2, 3] *= torch_a[:, :2, 3]
    test()

    # gradient flow through the advanced integer indexing operation
    kitty_b = kitty_a + 0
    kitty_b[[0, 0, 0], ..., [2, 2, 3]] *= kitty_a[[1, 1, -1], ..., [4, 3, 2]]
    torch_b = torch_a + 0
    torch_b[[0, 0, 0], ..., [2, 2, 3]] *= torch_a[[1, 1, -1], ..., [4, 3, 2]]
    test()

    # gradient flow through the advanced boolean indexing operation
    kitty_b = kitty_a.abs()
    kitty_mask = kitty_b.detach() ** 2 < 0.5
    kitty_b[kitty_mask] **= kitty_a[kitty_mask].abs()

    torch_b = torch_a.abs()
    torch_mask = torch_b.detach() ** 2 < 0.5
    torch_b[torch_mask] **= torch_a[torch_mask].abs()

    test()
