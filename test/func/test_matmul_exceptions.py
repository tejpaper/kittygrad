from test.conftest import *


def test_matmul_exceptions():
    # mm
    (kitty_a, kitty_b), _ = init_tensors([(3,), (3, 4)], [np.float64, np.float16])
    with pytest.raises(TypeError) as msg:
        kitty.mm(input=kitty_a)
    assert str(msg.value) == "missing a required argument: 'mat2'"

    with pytest.raises(TypeError) as msg:
        kitty.mm(kitty_a, 1)
    assert str(msg.value) == "Unsupported argument type(s) for mm: 'Tensor' and 'int'."

    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "2D tensors expected, but got 1D and 2D tensors."

    (kitty_a, kitty_b), _ = init_tensors([(2, 3), (3, 4, 1, 1)], [np.float32, np.float16])
    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "2D tensors expected, but got 2D and 4D tensors."

    (kitty_a, kitty_b), _ = init_tensors([(2, 3), (4, 3)], [np.float64, np.float32])
    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "input and mat2 shapes cannot be multiplied (2x3 and 4x3)."

    # dot
    (kitty_a, kitty_b), _ = init_tensors([(16, 16), (16,)], [np.float32, np.float64])
    with pytest.raises(TypeError) as msg:
        kitty.dot(kitty_a)
    assert str(msg.value) == "missing a required argument: 'other'"

    with pytest.raises(RuntimeError) as msg:
        kitty.dot(kitty_a, kitty_b)
    assert str(msg.value) == "1D tensors expected, but got 2D and 1D tensors."

    (kitty_a, kitty_b), _ = init_tensors([(18,), (16,)], [np.float16, np.float16])
    with pytest.raises(RuntimeError) as msg:
        kitty.dot(kitty_a, kitty_b)
    assert str(msg.value) == ("Inconsistent tensor size, expected tensor input and other to have "
                              "the same number of elements, but got 18 and 16 elements respectively.")

    # mv
    (kitty_a, kitty_b), _ = init_tensors([(2, 3, 4), (4,)])
    with pytest.raises(TypeError) as msg:
        kitty.mv(kitty_a, input=kitty_a)
    assert str(msg.value) == "multiple values for argument 'input'"

    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "input must be a matrix, not a 3D tensor."

    (kitty_a, kitty_b), _ = init_tensors([(3, 4), (4, 1)], [np.float64, np.float64])
    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "vec must be a vector, not a 2D tensor."

    (kitty_a, kitty_b), _ = init_tensors([(3, 4), (5,)], [np.float16, np.float32])
    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "input and vec shapes cannot be multiplied (3x4 and 5)."

    # bmm
    (kitty_a, kitty_b), _ = init_tensors([(2, 3, 4), (1, 2, 4, 2)], [np.float16, np.float64])
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("Batch dimensions of both tensors must be equal, but got "
                              "(2,) and (1, 2) respectively.")

    (kitty_a, kitty_b), _ = init_tensors([(2, 3, 4), (4, 2)], [np.float32, np.float16])
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("Batch dimensions of both tensors must be equal, but got "
                              "(2,) and () respectively.")

    (kitty_a, kitty_b), _ = init_tensors([(4,), (4, 2)], [np.float32, np.float64])
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("The batch matrix-matrix product requires the "
                              "tensors to have at least 3 dimensions each.")

    (kitty_a, kitty_b), _ = init_tensors([(13, 2, 3, 5), (13, 2, 4, 2)])
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == "input and mat2 matrix shapes cannot be multiplied (3x5 and 4x2)."

    # matmul
    with pytest.raises(TypeError) as msg:
        kitty.matmul()
    assert str(msg.value) == "missing a required argument: 'input'"

    with pytest.raises(RuntimeError) as msg:
        kitty.matmul(kitty.tensor(1), kitty.tensor([2, 3]))
    assert str(msg.value) == "Input tensors must not be scalars."

    with pytest.raises(TypeError) as msg:
        kitty.matmul(kitty.tensor(1), 1.5)
    assert str(msg.value) == "Unsupported argument type(s) for matmul: 'Tensor' and 'float'."
