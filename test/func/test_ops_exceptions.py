from test.conftest import *


def test_ops_exceptions():
    kitty_a = kitty.tensor(1, requires_grad=True)

    with pytest.raises(TypeError) as msg:
        kitty_a + (1 + 1j)
    assert str(msg.value) == "Unsupported operand type(s) for +: 'Tensor' and 'complex'."

    with pytest.raises(TypeError) as msg:
        (1 + 1j) * kitty_a
    assert str(msg.value) == "Unsupported operand type(s) for *: 'complex' and 'Tensor'."

    with pytest.raises(TypeError) as msg:
        kitty_a + None
    assert str(msg.value) == "Unsupported operand type(s) for +: 'Tensor' and 'NoneType'."

    kitty_a + np.array(2, dtype=np.int8)  # it's ok, auto cast

    with pytest.raises(TypeError) as msg:
        kitty_a.type(np.int8)
    assert str(msg.value) == "Data type 'int8' is not supported."
