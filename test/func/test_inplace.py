from test.conftest import *


@pytest.mark.parametrize(
    'dtypes', [
        [np.float32, np.float32],
        [np.float64, np.float32],
        [np.float64, np.float64],
    ])
@pytest.mark.parametrize(
    'shapes,squeeze_dims', [
        ([(2, 3), (2, 3)], ()),
        ([(1,), (1,)], ()),
        ([(1,), (1,)], (0,)),
        ([(5, 3, 4, 1), (3, 1, 1)], ()),
    ])
def test_inplace(shapes, dtypes, squeeze_dims, compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shapes, dtypes, squeeze_dims=squeeze_dims)

    def zero_grad():
        for tensor in (kitty_a, kitty_b, torch_a, torch_b):
            tensor.grad = None

    def test():
        assert compare(kitty_c, torch_c)
        assert compare(kitty_d, torch_d)

        kitty_c.retain_grad()
        torch_c.retain_grad()

        with pytest.warns(UserWarning, match="An attempt to assign a gradient to a tensor with retains_grad=True"):
            kitty_d.sum().backward()
        torch_d.sum().backward()

        assert compare(kitty_a.grad, torch_a.grad)
        assert compare(kitty_b.grad, torch_b.grad)
        assert compare(kitty_c.grad, torch_c.grad)

        zero_grad()

    # __iadd__
    kitty_c = kitty_a + 0
    kitty_c += kitty_c
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a + 0
    torch_c += torch_c
    torch_d = torch_c * torch_b

    test()

    # __isub__
    kitty_c = kitty_a + 0
    kitty_c -= kitty_b
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a + 0
    torch_c -= torch_b
    torch_d = torch_c * torch_b

    test()

    # __imul__
    kitty_c = kitty_a + 0
    kitty_c *= kitty_b
    kitty_d = 1 / kitty_c

    torch_c = torch_a + 0
    torch_c *= torch_b
    torch_d = 1 / torch_c

    assert compare(kitty_a, torch_a)
    test()

    # __ipow__
    kitty_c = kitty_a + 0
    kitty_c **= 2
    kitty_c **= 0.5
    kitty_c **= -kitty_c
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a + 0
    torch_c **= 2
    torch_c **= 0.5
    torch_c = torch_c ** -torch_c
    torch_d = torch_c * torch_b

    test()

    # __itruediv__, detach
    kitty_c = kitty_a.detach()
    kitty_c /= kitty_b
    kitty_d = kitty_c + kitty_b

    torch_c = torch_a.detach().clone()
    torch_c /= torch_b
    torch_d = torch_c + torch_b

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    torch_d.sum().backward()

    assert compare(kitty_b.grad, torch_b.grad)

    zero_grad()

    # clone, __iadd__, __imul__
    kitty_c = kitty_a + 0
    kitty_d = kitty_c.clone()
    kitty_c += kitty_b
    kitty_d *= kitty_b
    kitty_e = kitty_c + kitty_d

    torch_c = torch_a + 0
    torch_d = torch_c.clone()
    torch_c += torch_b
    torch_d *= torch_b
    torch_e = torch_c + torch_d

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)
    assert compare(kitty_e, torch_e)

    kitty_e.sum().backward()
    torch_e.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
