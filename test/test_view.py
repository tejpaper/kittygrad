from conftest import *


@pytest.mark.parametrize(
    'shape', [
        (4, 1, 3, 2, 1, 5),
        (1, 2, 3, 4, 1, 6),
    ])
def test_view(shape, compare):
    print()
    kitty_a, torch_a = map(next, init_tensors(shape))

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
    assert kitty_b is kitty_b

    kitty_b.sum().backward()
    torch_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)


def test_view_exceptions():
    pass


@pytest.mark.parametrize(
    'shape_a,shape_b,shape_c', [
        ((10, 1, 2), (3, 1), (2, 1, 1, 1)),
    ])
def test_broadcast_tensors(shape_a, shape_b, shape_c, compare):
    print()
    init_kitty, init_torch = init_tensors(shape_a, shape_b, shape_c)
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
