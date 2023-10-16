from test.conftest import *


def test_inplace_exceptions(compare):
    kitty_a = kitty.tensor([[1]])

    with pytest.raises(NotImplementedError) as msg:
        kitty_a @= kitty.tensor([[2]])
    assert str(msg.value) == "Inplace matrix multiplication is not implemented."

    with pytest.raises(TypeError) as msg:
        kitty_a **= 'string'
    assert str(msg.value) == "Unsupported operand type(s) for **=: 'Tensor' and 'str'."

    kitty_a[0, 0] += 1

    kitty_a.requires_grad = True

    with pytest.raises(RuntimeError) as msg:
        kitty_a /= 2
    assert str(msg.value) == "A leaf Variable that requires grad is being used in an in-place operation."

    with pytest.raises(RuntimeError) as msg:
        kitty_a[0] += 1
    assert str(msg.value) == "A leaf Variable that requires grad is being used in an in-place operation."

    kitty_b = kitty_a.expand(2, 3)

    with pytest.raises(RuntimeError) as msg:
        kitty_b += 1
    assert str(msg.value) == ("The inplace operation cannot be applied to a read-only tensor. If this "
                              "tensor is a view of another, you can try to do the same operation with it.")

    kitty_a = kitty.tensor([[2], [2]], dtype=kitty.float16)
    kitty_b = kitty.tensor([[2], [2]], requires_grad=True)

    with pytest.raises(RuntimeError) as msg:
        kitty_a *= kitty_b
    assert str(msg.value) == "Output with dtype 'float16' doesn't match the promotion dtype 'float32'."

    kitty_a = kitty.tensor([3, 3, 3])

    with pytest.raises(RuntimeError) as msg:
        kitty_a **= kitty_b
    assert str(msg.value) == "Output with shape (3,) doesn't match the broadcast shape (2, 3)."

    # converting a leaf tensor to a non-leaf tensor using inplace operation
    kitty_a = kitty.tensor(2., requires_grad=False)
    kitty_b = kitty.tensor(3., requires_grad=True)
    kitty_a *= kitty_b
    kitty_a.backward()

    torch_a = torch.tensor(2., requires_grad=False)
    torch_b = torch.tensor(3., requires_grad=True)
    torch_a *= torch_b
    torch_a.backward()

    assert compare(kitty_a, torch_a)
    assert compare(kitty_b.grad, torch_b.grad)

    with pytest.warns(UserWarning,
                      match="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed."):
        _ = kitty_a.grad
