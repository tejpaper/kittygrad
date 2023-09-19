from conftest import *


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32],
        [np.float64],
    ])
@pytest.mark.parametrize(
    'shapes,squeeze_dims', [
        ([(2, 3, 4, 5)], ()),
        ([(2, 1, 8, 1)], ()),
        ([(1,)], (0,)),
    ])
def test_activation(shapes, dtypes, squeeze_dims, compare):
    kitty_a, torch_a = map(next, init_tensors(shapes, dtypes, squeeze_dims=squeeze_dims))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    # sigmoid
    kitty_b = kitty_a.sigmoid()
    torch_b = torch_a.sigmoid()
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # tanh
    kitty_b = kitty_a.tanh()
    torch_b = torch_a.tanh()
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # relu
    kitty_b = kitty_a.relu()
    torch_b = torch_a.relu()
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # TODO: LeakyReLU, SiLU
