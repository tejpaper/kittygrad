from test.conftest import *


def test_view_exceptions():
    # transpose
    kitty_a = kitty.tensor(1, requires_grad=True)
    with pytest.raises(RuntimeError) as msg:
        kitty_a.transpose(0, 0)
    assert str(msg.value) == "Scalar cannot be transposed."

    kitty_a, _ = map(next, init_tensors([(2, 1, 3)]))
    with pytest.raises(IndexError) as msg:
        kitty_a.transpose(3, 0)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-3, 2], but got 3)."

    with pytest.raises(IndexError) as msg:
        kitty_a.transpose(1, -4)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-3, 2], but got -4)."

    # mT
    with pytest.raises(RuntimeError) as msg:
        _ = kitty.tensor(1, requires_grad=True).mT
    assert str(msg.value) == "tensor.mT is only supported on matrices or batches of matrices. Got 0D tensor."

    with pytest.raises(RuntimeError) as msg:
        _ = kitty.tensor([1, 2], requires_grad=True).mT
    assert str(msg.value) == "tensor.mT is only supported on matrices or batches of matrices. Got 1D tensor."

    # permute
    kitty_a, _ = map(next, init_tensors([(2, 4, 1, 3)], [np.float16]))
    with pytest.raises(RuntimeError) as msg:
        kitty_a.permute((0, 1, 2, 3, 4))
    assert str(msg.value) == ("Number of dimensions in the tensor input does not match "
                              "the length of the desired ordering of dimensions i.e. "
                              "input.dim() = 4 is not equal to len(dims) = 5.")

    with pytest.raises(RuntimeError) as msg:
        kitty_a.permute((1, 0, 3, 3))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.permute((1, 0, 20, 3))
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 20)."

    # squeeze
    with pytest.raises(RuntimeError) as msg:
        kitty_a.squeeze((0, -4))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.squeeze(16)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 16)."

    # unsqueeze
    with pytest.raises(IndexError) as msg:
        kitty_a.unsqueeze((-9, 1, 2, 7, 9))
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-9, 8], but got 9)."

    # expand, broadcast_to
    with pytest.raises(RuntimeError) as msg:
        kitty_a.expand(-1, 0, -1, -1)
    assert str(msg.value) == "The expanded size of the tensor (0) isn't allowed."

    with pytest.raises(RuntimeError) as msg:
        kitty_a.expand(13, -1, 2, 4, 1, 3)
    assert str(msg.value) == ("The expanded size of the tensor (-1) isn't allowed "
                              "in a leading, non-existing dimension 1.")

    with pytest.raises(RuntimeError) as msg:
        kitty.broadcast_to(kitty_a, [-1, -1, 3, 21])
    assert str(msg.value) == ("The expanded size of the tensor (21) must match the existing size (3) at non-singleton "
                              "dimension 3. Target sizes: (-1, -1, 3, 21). Tensor sizes: (2, 4, 1, 3).")
