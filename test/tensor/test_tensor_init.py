from kittygrad.utils.constants import DEFAULT_DTYPE
from test.conftest import *


def test_tensor_init():
    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=int)
    assert str(msg.value) == "Data type 'int' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=float)
    assert str(msg.value) == "Data type 'float' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=np.int64)
    assert str(msg.value) == f"Data type '{np.int64.__name__}' is not supported."

    with pytest.raises(TypeError) as msg:
        kitty.tensor(1, dtype=np.longlong)
    assert str(msg.value) == f"Data type '{np.longlong.__name__}' is not supported."

    with pytest.raises(RuntimeError) as msg:
        kitty.tensor(kitty.tensor([1, 1]))
    assert str(msg.value) == ("If you want to create a new tensor from another, use "
                              "source_tensor.detach() and then specify the requires_grad attribute.")

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
