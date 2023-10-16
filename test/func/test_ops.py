import itertools

from test.conftest import *


@pytest.mark.parametrize(
    'dtypes',
    itertools.product((np.float32, np.float64), (np.float32, np.float64)))
@pytest.mark.parametrize(
    'shapes,squeeze_dims', [
        ([(2, 3), (2, 3)], ()),
        ([(4, 1), (3,)], ()),
        ([(13, 1, 3), (2, 3)], ()),
        ([(1,), (1,)], ()),
        ([(1,), (1,)], (0,)),
        ([(5, 3, 4, 1), (3, 1, 1)], ()),
        ([(5, 1, 4, 1), (8,)], ()),
    ])
def test_ops(shapes, dtypes, squeeze_dims, compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shapes, dtypes, squeeze_dims=squeeze_dims)
    numpy_a = kitty_a._data
    numpy_b = kitty_b._data

    def zero_grad():
        for tensor in (kitty_a, kitty_b, torch_a, torch_b):
            tensor.grad = None

    # type, __mul__
    kitty_c = kitty_a.type(kitty.float64)
    kitty_c.retain_grad()
    (kitty_c * kitty_b.type(kitty.float64)).sum().backward()

    torch_c = torch_a.type(torch.float64)
    torch_c.retain_grad()
    (torch_c * torch_b.type(torch.float64)).sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)

    zero_grad()

    # __pos__, __neg__, ones_like
    kitty_c = +(-kitty_a)
    torch_c = +(-torch_a)
    assert compare(kitty_c, torch_c)

    kitty_c.backward(kitty.ones_like(kitty_c))
    torch_c.backward(torch.ones_like(torch_c))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # exp, sum
    kitty_c = kitty_a.exp()
    torch_c = torch_a.exp()
    assert compare(kitty_c, torch_c)

    kitty_c.sum().backward()
    torch_c.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # __pow__, log, mean
    kitty_c = (kitty_a ** 2).log()
    torch_c = (torch_a ** 2).log()
    assert compare(kitty_c, torch_c)

    kitty_c.mean().backward()
    torch_c.mean().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # __add__, __mul__, __rsub__, mean, retain_grad
    kitty_c = kitty_a + kitty_b
    kitty_d = kitty_c * kitty_b
    kitty_e = (2 - kitty_d).mean()

    torch_c = torch_a + torch_b
    torch_d = torch_c * torch_b
    torch_e = (2 - torch_d).mean()

    for kitty_t, torch_t in zip((kitty_c, kitty_d, kitty_e), (torch_c, torch_d, torch_e)):
        kitty_t.retain_grad()
        torch_t.retain_grad()
        assert compare(kitty_t, torch_t)

    kitty_e.backward()
    torch_e.backward()

    for kitty_t, torch_t in zip((kitty_a, kitty_b, kitty_c, kitty_d, kitty_e),
                                (torch_a, torch_b, torch_c, torch_d, torch_e)):
        assert compare(kitty_t.grad, torch_t.grad)

    zero_grad()

    # __truediv__, __rtruediv__, __add__, sum
    kitty_c = kitty_a / numpy_b
    kitty_d = 1 / kitty_b
    kitty_e = kitty_c + kitty_d

    torch_c = torch_a / torch_b.detach()
    torch_d = 1 / torch_b
    torch_e = torch_c + torch_d

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)
    assert compare(kitty_e, torch_e)

    kitty_e.sum().backward()
    torch_e.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)

    zero_grad()

    # __rpow__, __radd__, __rmul__, __sub__, mean
    kitty_c = 21 + 0.5 ** (kitty_b - 0.5 * kitty_b)
    torch_c = 21 + 0.5 ** (torch_b - 0.5 * torch_b)
    assert compare(kitty_c, torch_c)

    kitty_c.mean().backward()
    torch_c.mean().backward()
    assert compare(kitty_b.grad, torch_b.grad)

    zero_grad()

    # __add__, __mul__, sum, retain_grad
    kitty_c = kitty_a + numpy_b
    kitty_d = numpy_a * kitty_b
    kitty_e = kitty_c * kitty_d

    torch_c = torch_a + torch_b.detach()
    torch_d = torch_a.detach() * torch_b
    torch_e = torch_c * torch_d

    for kitty_t, torch_t in zip((kitty_c, kitty_d, kitty_e), (torch_c, torch_d, torch_e)):
        kitty_t.retain_grad()
        torch_t.retain_grad()
        assert compare(kitty_t, torch_t)

    kitty_e.sum().backward()
    torch_e.sum().backward()

    for kitty_t, torch_t in zip((kitty_a, kitty_b, kitty_c, kitty_d, kitty_e),
                                (torch_a, torch_b, torch_c, torch_d, torch_e)):
        assert compare(kitty_t.grad, torch_t.grad)

    zero_grad()

    # __abs__, __pow__, sum
    kitty_c = abs(kitty_a)
    kitty_d = kitty_c ** -0.5
    kitty_d.sum().backward()

    torch_c = abs(torch_a)
    torch_d = torch_c ** -0.5
    torch_d.sum().backward()

    assert compare(kitty_d, torch_d)
    assert compare(kitty_a.grad, torch_a.grad)
