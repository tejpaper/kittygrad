import itertools

from test.conftest import *


@pytest.mark.parametrize(
    'dtypes',
    itertools.product((np.float32, np.float64), (np.float32, np.float64)))
@pytest.mark.parametrize(
    'shapes', [
        [(4,), (4,)],  # dot
        [(2, 3), (3, 4)],  # mm
        [(5,), (5, 2)],  # mm, squeeze, unsqueeze
        [(5, 2), (2,)],  # mv
        [(4,), (2, 4, 2)],  # bmm, squeeze, unsqueeze, expand
        [(2, 2, 4), (4,)],  # bmm, squeeze, unsqueeze, expand
        [(3, 4), (2, 4, 2)],  # bmm, expand
        [(2, 2, 4), (4, 3)],  # bmm, expand
        [(5, 3, 4), (5, 4, 2)],  # bmm
    ])
def test_matmul(shapes, dtypes, compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shapes, dtypes)

    kitty_c = kitty_a @ kitty_b
    result_type = torch.result_type(torch_a, torch_b)
    torch_c = torch_a.type(result_type) @ torch_b.type(result_type)
    assert compare(kitty_c, torch_c)

    kitty_c.sum().backward()
    torch_c.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
