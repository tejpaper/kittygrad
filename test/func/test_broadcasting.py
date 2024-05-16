import itertools

from kittygrad.core import ALL_DTYPES
from test.conftest import *


@pytest.mark.parametrize(
    'dtypes',
    itertools.product(ALL_DTYPES, ALL_DTYPES, ALL_DTYPES))
@pytest.mark.parametrize(
    'shapes', [
        [(10, 1, 2), (3, 1), (2, 1, 1, 1)],
    ])
def test_broadcasting(shapes, dtypes, compare):
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

    kitty_a.grad = None
    torch_a.grad = None

    # kwargs input
    kitty_a_view = kitty.broadcast_to(input=kitty_a, shape=(-1, 3, -1))
    torch_a_view = torch.broadcast_to(torch_a, (-1, 3, -1))
    assert compare(kitty_a_view, torch_a_view)

    kitty_a_view.sum().backward()
    torch_a_view.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)
