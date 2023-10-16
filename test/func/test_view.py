from test.conftest import *


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
