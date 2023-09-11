import pytest
import torch
import numpy as np
import kittygrad as kitty


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


def test_exceptions():
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

    tensor = kitty.tensor([1, 1])  # kitty.float32

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
    tensor.requires_grad = False
    tensor.grad = None


def test_arithmetics():  # TODO: not only exceptions, but also correctness of the results
    tensor = kitty.tensor(3)

    tensor + 1
    1 + tensor
    tensor + 1.
    1. + tensor

    with pytest.raises(TypeError) as msg:
        tensor + (1 + 1j)
    assert str(msg.value) == "Unsupported operand type(s) for +: 'Tensor' and 'complex'."

    with pytest.raises(TypeError) as msg:
        (1 + 1j) * tensor
    assert str(msg.value) == "Unsupported operand type(s) for *: 'complex' and 'Tensor'."

    with pytest.raises(TypeError) as msg:
        tensor + None
    assert str(msg.value) == "Unsupported operand type(s) for +: 'Tensor' and 'NoneType'."

    with pytest.raises(NotImplementedError) as msg:
        np.array(2, dtype=np.int8) + tensor
    assert str(msg.value) == "Unsupported operation with NumPy array. Try swapping operands."
    tensor + np.array(2, dtype=np.int8)  # it's ok, auto cast

    with pytest.raises(TypeError) as msg:
        kitty.tensor(0, dtype=kitty.half) * tensor
    assert str(msg.value) == "Operands type mismatch: float16 != float32."


def test_indexing():
    torch_a = torch.tensor([1, 2, 3.])
    torch_b = torch_a
    torch_b[1] = 100

    kitty_a = kitty.tensor([1, 2, 3.])
    kitty_b = kitty_a
    kitty_b[1] = 100

    assert repr(kitty_a) == repr(torch_a)

    torch_a = torch.tensor([1., 2, 3])
    torch_b = torch_a[:2]

    kitty_a = kitty.tensor([1., 2, 3])
    kitty_b = kitty_a[:2]

    assert repr(kitty_b) == repr(torch_b)

    torch_b[0] = 1000
    kitty_b[0] = 1000

    assert repr(kitty_a) == repr(torch_a)

    torch_a = torch.tensor([1, 2, 3.])
    torch_a.data[0] = 100

    kitty_a = kitty.tensor([1, 2, 3.])
    kitty_a.data[0] = 100

    assert repr(kitty_a) == repr(torch_a)

    torch_a = torch.tensor([[1, 2, 3], [4, 5, 6.]])
    kitty_a = kitty.tensor([[1, 2, 3], [4, 5, 6.]])

    assert repr(kitty_a[:, ::2]) == repr(torch_a[:, ::2])

    torch_a = torch.tensor([1, 2.])
    torch_b = torch.tensor([[0, 0.], [0, 0], [0, 0]])
    torch_b[1] = torch_a

    kitty_a = kitty.tensor([1, 2.])
    kitty_b = kitty.tensor([[0, 0.], [0, 0], [0, 0]])
    kitty_b[1] = kitty_a

    assert repr(kitty_b) == repr(torch_b)

    torch_a[0] = 1000
    kitty_a[0] = 1000

    assert repr(kitty_b) == repr(torch_b)


def test_ref():
    kitty_tensor = kitty.tensor([1, 2, 3], dtype=kitty.double)
    assert kitty_tensor.dtype == kitty.double

    kitty_detached = kitty_tensor.detach()
    kitty_detached[1] = 1000

    assert kitty_tensor != kitty_detached
