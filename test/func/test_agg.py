from test.conftest import *


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32],
        [np.float64],
    ])
@pytest.mark.parametrize(
    'shapes', [
        [(2, 3, 4, 5)],
        [(2, 1, 8, 1)],
    ])
def test_agg(shapes, dtypes, compare):
    kitty_a, torch_a = map(next, init_tensors(shapes, dtypes))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    # sum : dim=tuple, keepdim=True
    kitty_b = kitty_a.sum(dim=(3, -3), keepdim=True)
    torch_b = torch_a.sum(dim=(3, -3), keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # var : dim=list, keepdim=False
    kitty_b = kitty_a.var(dim=[-1, 0], correction=0, keepdim=False)
    torch_b = torch_a.var(dim=[-1, 0], correction=0, keepdim=False)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # sum : dim=None, keepdim=True
    kitty_b = kitty_a.sum(keepdim=True)
    torch_b = torch_a.sum(dim=None, keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # sum : dim is empty tuple
    kitty_b = kitty_a.sum(dim=())
    torch_b = torch_a + 0  # TODO: ref [2]
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    # std : dim=int
    kitty_b = kitty_a.std(dim=-2)
    torch_b = torch_a.std(dim=-2)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)
