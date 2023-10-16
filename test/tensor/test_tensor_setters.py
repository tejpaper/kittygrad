from test.conftest import *


def test_tensor_setters():
    tensor = kitty.tensor([1, 1])

    with pytest.raises(TypeError) as msg:
        tensor.grad = 0
    assert str(msg.value) == "Assigned grad expected to be a Tensor or None but got grad of 'int'."

    with pytest.raises(TypeError) as msg:
        tensor.grad = np.array([0, 0], dtype=np.float32)
    assert str(msg.value) == "Assigned grad expected to be a Tensor or None but got grad of 'ndarray'."

    with pytest.raises(RuntimeError) as msg:
        tensor.grad = tensor
    assert str(msg.value) == "Can't assign Variable as its own grad."

    with pytest.raises(TypeError) as msg:
        tensor.grad = kitty.tensor([-1, 0], dtype=kitty.double)
    assert str(msg.value) == "Assigned grad has data of a different type."

    with pytest.raises(RuntimeError) as msg:
        tensor.grad = kitty.tensor([[-1, 0]])
    assert str(msg.value) == "Assigned grad has data of a different size."

    with pytest.warns(UserWarning, match="Trying to assign a gradient to a tensor that doesn't need it."):
        tensor.grad = kitty.tensor([0.9, -0.5], dtype=kitty.float)

    tensor.grad = None
    tensor.requires_grad = True

    with pytest.raises(RuntimeError) as msg:
        (tensor * 2).requires_grad = True
    assert str(msg.value) == "You can only change requires_grad flags of leaf variables."

    with pytest.raises(RuntimeError) as msg:
        (tensor * 2).requires_grad = False
    assert str(msg.value) == ("You can only change requires_grad flags of leaf variables."
                              "If you want to use a computed variable in a subgraph that doesn't "
                              "require differentiation use var_no_grad = var.detach().")
