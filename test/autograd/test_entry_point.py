from test.conftest import *


def test_entry_point(compare):
    kitty_a = kitty.tensor([2.], requires_grad=True)
    torch_a = torch.tensor([2.], requires_grad=True)

    # no backward graph at all
    kitty_a.requires_grad = False
    with pytest.raises(RuntimeError) as msg:
        (kitty_a * 2).sum().backward()
    assert str(msg.value) == "Tensor does not require grad and does not have a grad_fn."
    assert kitty_a.grad is None
    kitty_a.requires_grad = True

    # non-scalar implicit gradient initialization
    with pytest.raises(RuntimeError) as msg:
        kitty_a.backward()
    assert str(msg.value) == "Grad can be implicitly created only for scalar outputs."
    assert kitty_a.grad is None

    # wrong initial gradient size
    with pytest.raises(RuntimeError) as msg:
        (2 * kitty_a).backward(kitty.ones_like(kitty_a).unsqueeze(0))
    assert str(msg.value) == "Initial gradient has data of a different size."
    assert kitty_a.grad is None

    # gradient accumulation without a computational graph with wrong size
    with pytest.raises(RuntimeError) as msg:
        kitty_a.backward(kitty.ones_like(kitty_a).unsqueeze(0))
    assert str(msg.value) == "Assigned grad has data of a different size."
    assert kitty_a.grad is None

    # wrong initial gradient dtype
    with pytest.raises(TypeError) as msg:
        (kitty_a * 2).backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    assert str(msg.value) == "Initial gradient has data of a different type."
    assert kitty_a.grad is None

    # gradient accumulation without a computational graph with wrong dtype
    with pytest.raises(TypeError) as msg:
        kitty_a.backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    assert str(msg.value) == "Assigned grad has data of a different type."
    assert kitty_a.grad is None

    # gradient accumulation without a computational graph
    kitty_a.backward(kitty.ones_like(kitty_a))
    torch_a.backward(torch.ones_like(torch_a))
    assert compare(kitty_a.grad, torch_a.grad)
