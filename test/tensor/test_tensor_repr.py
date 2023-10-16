from test.conftest import *


def test_tensor_repr():
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
