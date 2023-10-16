from test.conftest import *


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32],
        [np.float64],
    ])
@pytest.mark.parametrize(
    'shapes', [
        [(2, 3, 4, 5)],
        [(2, 1, 8, 1)],
    ])
def test_agg_exceptions(shapes, dtypes):
    kitty_a, _ = map(next, init_tensors(shapes))

    with pytest.raises(RuntimeError) as msg:
        kitty_a.sum((0, 0))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(RuntimeError) as msg:
        kitty_a.mean((0, -4))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.std((42, 0))
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 42)."

    with pytest.raises(IndexError) as msg:
        kitty_a.sum(-5)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got -5)."
