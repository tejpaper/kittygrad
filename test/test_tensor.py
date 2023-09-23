from conftest import *
from kittygrad.utils import DEFAULT_DTYPE


def test_initialization():
    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=int)
    assert str(msg.value) == "Data type 'int' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=float)
    assert str(msg.value) == "Data type 'float' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=np.int64)
    assert str(msg.value) == "Data type 'int64' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=np.longlong)
    assert str(msg.value) == "Data type 'longlong' is not supported."

    with pytest.raises(RuntimeError) as msg:
        kitty.tensor(kitty.tensor([1, 1]))
    assert str(msg.value) == ("If you want to create a new tensor from another, use "
                              "sourceTensor.detach() and then specify the requires_grad attribute.")

    with pytest.warns(UserWarning, match="Passed NumPy array has an unsupported data type."):
        tensor = kitty.tensor(np.array([1, 1], dtype=np.int8))
    assert tensor.dtype == DEFAULT_DTYPE
    assert kitty.tensor(1).dtype == DEFAULT_DTYPE

    data = np.array([1, 1], dtype=np.int8)
    tensor = kitty.tensor(data, dtype=np.float32)
    assert tensor.dtype == kitty.float32

    data = np.array([1, 1], dtype=np.float32)
    tensor = kitty.tensor(data, dtype=kitty.float64)
    data[0] = 100
    assert tensor[0].item() == 1

    tensor = kitty.tensor(data, dtype=kitty.float32)
    data[0] = 200
    assert tensor[0].item() == 200


def test_repr():
    torch_tensor_1 = torch.tensor([
        [1, 2, 3],
        [4, 18, 31],
        [4, 8, 31],
        [4, 9, -1],
    ], dtype=torch.float32)
    kitty_tensor_1 = kitty.tensor([
        [1, 2, 3],
        [4, 18, 31],
        [4, 8, 31],
        [4, 9, -1],
    ], dtype=kitty.float32)

    assert str(kitty_tensor_1) == str(torch_tensor_1).replace('torch.', '')
    assert repr(kitty_tensor_1) == repr(torch_tensor_1).replace('torch.', '')

    torch_tensor_1.requires_grad = True
    kitty_tensor_1.requires_grad = True

    assert str(kitty_tensor_1) == str(torch_tensor_1).replace('torch.', '')
    assert repr(kitty_tensor_1) == repr(torch_tensor_1).replace('torch.', '')

    torch_tensor_2 = torch.tensor([[
        [[1, 1],
         [1, 1]],
        [[2, 2],
         [2, 2]],
        [[3, 3],
         [3, 3]],
        [[4, 4],
         [4, 4]],
    ]], dtype=torch.float16)
    kitty_tensor_2 = kitty.tensor([[
        [[1, 1],
         [1, 1]],
        [[2, 2],
         [2, 2]],
        [[3, 3],
         [3, 3]],
        [[4, 4],
         [4, 4]],
    ]], dtype=kitty.float16)

    assert str(kitty_tensor_2) == str(torch_tensor_2).replace('torch.', '')
    assert repr(kitty_tensor_2) == repr(torch_tensor_2).replace('torch.', '')

    torch_tensor_2.requires_grad = True
    kitty_tensor_2.requires_grad = True

    assert str(kitty_tensor_2) == str(torch_tensor_2).replace('torch.', '')
    assert repr(kitty_tensor_2) == repr(torch_tensor_2).replace('torch.', '')

    torch_tensor_3 = torch.tensor(1, dtype=torch.float64)
    kitty_tensor_3 = kitty.tensor(1, dtype=kitty.float64)

    assert str(kitty_tensor_3) == str(torch_tensor_3).replace('torch.', '')
    assert repr(kitty_tensor_3) == repr(torch_tensor_3).replace('torch.', '')

    torch_tensor_3.requires_grad = True
    kitty_tensor_3.requires_grad = True

    assert str(kitty_tensor_3) == str(torch_tensor_3).replace('torch.', '')
    assert repr(kitty_tensor_3) == repr(torch_tensor_3).replace('torch.', '')

    assert repr(kitty.tensor(1, dtype=kitty.half)) == repr(torch.tensor(1, dtype=torch.half)).replace('torch.', '')
    assert repr(kitty.tensor(1, dtype=kitty.float)) == repr(torch.tensor(1, dtype=torch.float)).replace('torch.', '')
    assert repr(kitty.tensor(1, dtype=kitty.double)) == repr(torch.tensor(1, dtype=torch.double)).replace('torch.', '')


def test_getters():
    kitty_a = kitty.tensor(1, requires_grad=True)
    assert kitty_a.grad is None
    assert kitty_a.grad_fn is None

    kitty_b = kitty_a * 2
    assert not kitty_b.is_leaf
    assert kitty_b.grad_fn is not None

    with pytest.warns(UserWarning,
                      match="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed."):
        kitty_b.grad

    assert not kitty_b.retains_grad
    kitty_b.retain_grad()
    assert kitty_b.retains_grad

    assert kitty_a.version == 0
    assert kitty_b.version == 0

    kitty_b.backward()
    assert kitty_b.grad.item() == 1


def test_setters():
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


def test_methods():
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


def test_create():
    pass
