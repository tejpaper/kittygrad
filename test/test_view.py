import itertools

from conftest import *
from kittygrad.utils import ALL_DTYPES


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32],
        [np.float64],
    ])
@pytest.mark.parametrize(
    'shapes', [
        [(4, 1, 3, 2, 1, 5)],
        [(1, 2, 3, 4, 1, 6)],
    ])
def test_view(shapes, dtypes, compare):
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

    # transpose
    kitty_b = 1 / (2 * kitty_a).transpose(-3, 2)
    torch_b = 1 / (2 * torch_a).transpose(-3, 2)
    test()

    # transpose the same dim
    kitty_b = 1 / (2 * kitty_a).transpose(1, -5)
    torch_b = 1 / (2 * torch_a).transpose(1, -5)
    test()

    # mT
    kitty_b = (kitty_a + 3).mT ** -1
    torch_b = (torch_a + 3).mT ** -1
    test()

    # permute
    kitty_b = -kitty_a.relu().permute((-1, 0, 3, -2, 1, -4))
    torch_b = -torch_a.relu().permute((-1, 0, 3, -2, 1, -4))
    test()

    # squeeze dim=None
    kitty_b = kitty_a.squeeze()
    torch_b = torch_a.squeeze()
    test()

    # squeeze dim=tuple
    kitty_b = kitty_a.squeeze((1, -6))
    torch_b = torch_a.squeeze((1, -6))
    test()

    # squeeze dim=list
    kitty_b = kitty_a.squeeze([-1, -2])
    torch_b = torch_a.squeeze([-1, -2])
    test()

    # squeeze dim=int
    kitty_b = kitty_a.squeeze(-2)
    torch_b = torch_a.squeeze(-2)
    test()

    # unsqueeze dim=tuple
    kitty_b = kitty_a.unsqueeze((0, 7))
    torch_b = torch_a.unsqueeze(0).unsqueeze(7)
    test()

    # unsqueeze dim=list
    kitty_b = kitty_a.unsqueeze([-1, -2])
    torch_b = torch_a.unsqueeze(-1).unsqueeze(-1)
    test()

    # unsqueeze dim=int
    kitty_b = kitty_a.unsqueeze(4)
    torch_b = torch_a.unsqueeze(4)
    test()

    # expand
    kitty_b = kitty_a.expand(13, -1, -1, 3, -1, 6, -1)
    torch_b = torch_a.expand(13, -1, -1, 3, -1, 6, -1)
    test()

    # expand without any impact
    kitty_b = kitty_a.expand(-1, -1, -1, -1, -1, -1)
    torch_b = torch_a.expand(-1, -1, -1, -1, -1, -1)
    assert kitty_b is kitty_a

    kitty_b.sum().backward()
    torch_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)


def test_view_exceptions():
    # transpose
    kitty_a = kitty.tensor(1, requires_grad=True)
    with pytest.raises(RuntimeError) as msg:
        kitty_a.transpose(0, 0)
    assert str(msg.value) == "Scalar cannot be transposed."

    kitty_a, _ = map(next, init_tensors([(2, 1, 3)]))
    with pytest.raises(IndexError) as msg:
        kitty_a.transpose(3, 0)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-3, 2], but got 3)."

    with pytest.raises(IndexError) as msg:
        kitty_a.transpose(1, -4)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-3, 2], but got -4)."

    # mT
    with pytest.raises(RuntimeError) as msg:
        kitty.tensor(1, requires_grad=True).mT
    assert str(msg.value) == "tensor.mT is only supported on matrices or batches of matrices. Got 0D tensor."

    with pytest.raises(RuntimeError) as msg:
        kitty.tensor([1, 2], requires_grad=True).mT
    assert str(msg.value) == "tensor.mT is only supported on matrices or batches of matrices. Got 1D tensor."

    # permute
    kitty_a, _ = map(next, init_tensors([(2, 4, 1, 3)], [np.float16]))
    with pytest.raises(RuntimeError) as msg:
        kitty_a.permute((0, 1, 2, 3, 4))
    assert str(msg.value) == ("Number of dimensions in the tensor input does not match "
                              "the length of the desired ordering of dimensions i.e. "
                              "input.dim() = 4 is not equal to len(dims) = 5.")

    with pytest.raises(RuntimeError) as msg:
        kitty_a.permute((1, 0, 3, 3))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.permute((1, 0, 20, 3))
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 20)."

    # squeeze
    with pytest.raises(RuntimeError) as msg:
        kitty_a.squeeze((0, -4))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.squeeze(16)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 16)."

    # unsqueeze
    with pytest.raises(IndexError) as msg:
        kitty_a.unsqueeze((-9, 1, 2, 7, 9))
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-9, 8], but got 9)."

    # expand, broadcast_to
    with pytest.raises(RuntimeError) as msg:
        kitty_a.expand(-1, 0, -1, -1)
    assert str(msg.value) == "The expanded size of the tensor (0) isn't allowed."

    with pytest.raises(RuntimeError) as msg:
        kitty_a.expand(13, -1, 2, 4, 1, 3)
    assert str(msg.value) == ("The expanded size of the tensor (-1) isn't allowed "
                              "in a leading, non-existing dimension 1.")

    with pytest.raises(RuntimeError) as msg:
        kitty.broadcast_to(kitty_a, [-1, -1, 3, 21])
    assert str(msg.value) == ("The expanded size of the tensor (21) must match the existing size (3) at non-singleton "
                              "dimension 3. Target sizes: (-1, -1, 3, 21). Tensor sizes: (2, 4, 1, 3).")


@pytest.mark.parametrize(
    'dtypes',
    itertools.product(ALL_DTYPES, ALL_DTYPES, ALL_DTYPES))
@pytest.mark.parametrize(
    'shapes', [
        [(10, 1, 2), (3, 1), (2, 1, 1, 1)],
    ])
def test_broadcast_tensors(shapes, dtypes, compare):
    init_kitty, init_torch = init_tensors(shapes, dtypes)
    kitty_a, kitty_b, kitty_c = init_kitty
    torch_a, torch_b, torch_c = init_torch

    kitty_b.requires_grad = False
    torch_b.requires_grad = False

    kitty_a_view, kitty_b_view, kitty_c_view = kitty.broadcast_tensors(kitty_a, kitty_b, kitty_c)
    torch_a_view, torch_b_view, torch_c_view = torch.broadcast_tensors(torch_a, torch_b, torch_c)

    assert compare(kitty_a_view, torch_a_view)
    assert compare(kitty_b_view, torch_b_view)
    assert compare(kitty_c_view, torch_c_view)

    kitty_a_view.sum().backward()
    torch_a_view.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    with pytest.raises(RuntimeError) as msg:
        kitty_b_view.sum().backward()
    assert str(msg.value) == "Tensor does not require grad and does not have a grad_fn."
    assert kitty_b.grad is None

    kitty_c_view.sum().backward()
    torch_c_view.sum().backward()
    assert compare(kitty_c.grad, torch_c.grad)


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

