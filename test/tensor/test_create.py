from kittygrad.utils import DEFAULT_DTYPE
from test.conftest import *


def test_create(compare):
    assert compare(kitty.rand(2, 3) * 0, torch.rand(2, 3) * 0)

    kitty_a = kitty.randn(2, 3, requires_grad=True)
    torch_a = torch.randn(2, 3, requires_grad=True)

    assert compare(kitty_a * 0, torch_a * 0)
    assert compare(kitty.ones(2, 3), torch.ones(2, 3))
    assert compare(kitty.ones_like(kitty_a, requires_grad=True), torch.ones_like(torch_a, requires_grad=True))
    assert compare(kitty.zeros(2, 3, requires_grad=True), torch.zeros(2, 3, requires_grad=True))
    assert compare(kitty.zeros_like(kitty_a), torch.zeros_like(torch_a))

    with pytest.warns(UserWarning, match="Passed NumPy array has an unsupported data type."):
        tensor = kitty.ones(2, dtype=np.int32)
    assert tensor.dtype == DEFAULT_DTYPE
