from conftest import *


@pytest.mark.parametrize(
    'shape_a,shape_b,squeeze_dims', [
        ((2, 3), (2, 3), ()),
        ((4, 1), (3,), ()),
        ((13, 1, 3), (2, 3), ()),
        ((1,), (1,), ()),
        ((1,), (1,), (0,)),
        ((5, 3, 4, 1), (3, 1, 1), ()),
        ((5, 1, 4, 1), (8,), ()),
    ])
def test_ops(shape_a, shape_b, squeeze_dims, compare):
    print()
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shape_a, shape_b, squeeze_dims=squeeze_dims)

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
    kitty_c = kitty_a / 10
    kitty_d = 1 / kitty_b
    kitty_e = kitty_c + kitty_d

    torch_c = torch_a / 10
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
    kitty_c = kitty_a + kitty_b
    kitty_d = kitty_a * kitty_b
    kitty_e = kitty_c * kitty_d

    torch_c = torch_a + torch_b
    torch_d = torch_a * torch_b
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

    # TODO: abs + raising to a negative power
    # TODO: std
    # TODO: init grad with zeros, other non-standard variations


def test_ops_exceptions():
    pass


@pytest.mark.parametrize(
    'shape', [
        (2, 3, 4, 5),
        (2, 1, 8, 1),
    ])
def test_agg(shape, compare):
    print()
    kitty_a, torch_a = map(next, init_tensors(shape))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    # sum : dim=tuple, keepdim=True
    kitty_b = kitty_a.sum(dim=(3, -3), keepdim=True)
    torch_b = torch_a.sum(dim=(3, -3), keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # mean : dim=list, keepdim=False   # TODO: replace with std
    kitty_b = kitty_a.mean(dim=[-1, 0], keepdim=False)
    torch_b = torch_a.mean(dim=[-1, 0], keepdim=False)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # sum : dim=None, keepdim=True
    kitty_b = kitty_a.sum(keepdim=True)
    torch_b = torch_a.sum(dim=None, keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # sum : dim is empty tuple
    kitty_b = kitty_a.sum(dim=())
    torch_b = torch_a + 0  # TODO: ref [2]
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    # sum : dim=int
    kitty_b = kitty_a.sum(dim=-1)
    torch_b = torch_a.sum(dim=-1)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)


def test_agg_exceptions():
    pass


@pytest.mark.parametrize(
    'shape_a,shape_b', [
        ((4,), (4,)),  # dot
        ((2, 3), (3, 4)),  # mm
        ((5,), (5, 2)),  # mm, squeeze, unsqueeze
        ((5, 2), (2,)),  # mv
        ((4,), (2, 4, 2)),  # bmm, squeeze, unsqueeze, expand
        ((2, 2, 4), (4,)),  # bmm, squeeze, unsqueeze, expand
        ((3, 4), (2, 4, 2)),  # bmm, expand
        ((2, 2, 4), (4, 3)),  # bmm, expand
        ((5, 3, 4), (5, 4, 2)),  # bmm
    ])
def test_matmul(shape_a, shape_b, compare):
    print()
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shape_a, shape_b)

    kitty_c = kitty_a @ kitty_b
    torch_c = torch_a @ torch_b
    assert compare(kitty_c, torch_c)

    kitty_c.sum().backward()
    torch_c.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)


def test_matmul_exceptions():
    pass


def test_inplace():
    pass


def test_inplace_exceptions():
    pass
