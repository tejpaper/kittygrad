from test.conftest import *


def test_tensor_methods():
    tensor = kitty.tensor([1, 2, 3], dtype=kitty.double)
    assert tensor.dtype == kitty.double

    detached = tensor.detach()
    detached[1] = 1000

    assert tensor is not detached

    assert tensor.type(kitty.float64) is tensor
    assert tensor.type(kitty.float16).dtype == kitty.float16
    assert tensor.type(kitty.float32).dtype == kitty.float32

    with pytest.raises(RuntimeError) as msg:
        tensor.item()
    assert str(msg.value) == "A Tensor with 3 elements cannot be converted to Scalar."

    with pytest.raises(RuntimeError) as msg:
        tensor.retain_grad()
    assert str(msg.value) == "Can't retain_grad on Tensor that has requires_grad=False."

    tensor.requires_grad = True
    tensor.retain_grad()
    assert not tensor.retains_grad
