from test.conftest import *


def test_tensor_getters():
    kitty_a = kitty.tensor(1, requires_grad=True)
    assert kitty_a.grad is None
    assert kitty_a.grad_fn is None

    kitty_b = kitty_a * 2
    assert not kitty_b.is_leaf
    assert kitty_b.grad_fn is not None

    with pytest.warns(UserWarning,
                      match="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed."):
        _ = kitty_b.grad

    assert not kitty_b.retains_grad
    kitty_b.retain_grad()
    assert kitty_b.retains_grad

    assert kitty_a.version == 0
    assert kitty_b.version == 0

    kitty_b.backward()
    assert kitty_b.grad.item() == 1
