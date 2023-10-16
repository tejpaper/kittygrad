from test.conftest import *


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32],
        [np.float64],
    ])
@pytest.mark.parametrize(
    'shapes', [
        [(1, 3, 4)],
        [(1,)],
    ])
def test_version_control(shapes, dtypes, compare):
    kitty_a, torch_a = map(next, init_tensors(shapes, dtypes))

    # warning about possible problems with .retain_grad()
    kitty_b = kitty_a * 2
    kitty_b.retain_grad()
    kitty_c = 3 * kitty_b
    kitty_b += 1
    kitty_d = (kitty_c + kitty_b).sum()

    with pytest.warns(UserWarning, match="An attempt to assign a gradient to a tensor with retains_grad=True"):
        kitty_d.backward()

    torch_b = torch_a * 2
    torch_b.retain_grad()
    torch_c = 3 * torch_b
    torch_b += 1
    torch_d = (torch_c + torch_b).sum()
    torch_d.backward()

    assert compare(kitty_d, torch_d)
    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)  # undefined behavior, not recommended

    # inplace modification error caused by changing saved tensor
    kitty_b = 2 * kitty_a
    kitty_c = kitty_b ** -1

    assert kitty_b.version == 0

    kitty_b /= 2

    assert kitty_b.version == 1
    assert kitty_c.grad_fn.next_functions[0]._ctx.out is kitty_b

    with pytest.raises(RuntimeError) as msg:
        kitty_c.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # inplace modification error caused by changing output tensor
    kitty_b = 2 ** kitty_a
    kitty_b += 1

    with pytest.raises(RuntimeError) as msg:
        kitty_b.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # correctness of view operations with respect to version and data changes
    kitty_a.requires_grad = False

    for i, (method, args) in enumerate([
        (kitty_a.transpose, (0, -1)),
        (kitty_a.permute, (range(kitty_a.ndim - 1, -1, -1),)),
        (kitty_a.squeeze, ()),
        (kitty_a.unsqueeze, (-1,)),
        (kitty_a.expand, (3, *kitty_a.shape)),
        (kitty_a.__getitem__, (0,)),
    ]):
        kitty_b = method(*args)

        assert kitty_a.version == i
        assert kitty_b._version is kitty_a._version

        kitty_a += 1

        assert kitty_b.version == i + 1
        assert kitty_b._data.base is kitty_a._data
