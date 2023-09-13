import pytest
import torch
import numpy as np
import kittygrad as kitty


class Comparison:
    def __init__(self, relative_tol: float = 1e-5, absolute_tol: float = 1e-8) -> None:
        self.exact = 0
        self.approximate = 0
        self.max_rel_diff = 0

        self.rel_tol = relative_tol
        self.abs_tol = absolute_tol

    def __call__(self, kitty_tensor: kitty.Tensor, torch_tensor: torch.Tensor) -> bool:
        for attr in ('requires_grad', 'is_leaf', 'retains_grad'):
            attr_value_1 = getattr(kitty_tensor, attr)
            attr_value_2 = getattr(torch_tensor, attr)

            if attr_value_1 != attr_value_2:
                print(f'{attr} attribute mismatch: {attr_value_1} != {attr_value_2}.')
                return False

        # TODO: check dtypes

        kitty_array = kitty_tensor._data
        torch_array = torch_tensor.detach().numpy()

        exact_match = np.array_equal(kitty_array, torch_array)

        if exact_match:
            self.exact += 1
            self.approximate += 1
            return True

        if kitty_array.shape != torch_array.shape:
            print(f'Shapes mismatch: {kitty_array.shape} != {torch_array.shape}.')
            return False

        approximate_match = np.allclose(kitty_array, torch_array, self.rel_tol, self.abs_tol)

        self.approximate += approximate_match
        self.max_rel_diff = max(self.max_rel_diff,
                                np.max(self.rel_tol - np.abs(kitty_array - torch_array) / np.abs(torch_array)))

        return approximate_match

    def __str__(self) -> str:
        return (f'Exact: {self.exact} | '
                f'approximate: {self.approximate} | '
                f'maximum relative difference: {self.max_rel_diff}')


@pytest.fixture
def compare():
    cmp = Comparison()
    yield cmp
    print(cmp)


def init_tensors(*shapes: kitty.Size, squeeze_dims: kitty.Size = None):
    if squeeze_dims is None:
        squeeze_dims = tuple()

    np.random.seed(42)
    data = [np.random.randn(*shape).astype(np.float32).squeeze(squeeze_dims) for shape in shapes]
    return map(lambda lib: map(lambda d: lib.tensor(d, requires_grad=True), data), (kitty, torch))


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


@pytest.mark.parametrize(
    'shape', [
        (2, 3, 4, 5),
        (2, 1, 8, 1),
    ])
def test_activation(shape, compare):
    print()
    kitty_a, torch_a = map(next, init_tensors(shape))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    # sigmoid, sum
    kitty_b = kitty_a.sigmoid().sum(dim=(3, -3), keepdim=True)
    torch_b = torch_a.sigmoid().sum(dim=(3, -3), keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # tanh, mean  # TODO: replace with std
    kitty_b = kitty_a.tanh().mean(dim=[-1, 0], keepdim=False)
    torch_b = torch_a.tanh().mean(dim=[-1, 0], keepdim=False)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # relu, sum
    kitty_b = kitty_a.relu().sum(keepdim=True)
    torch_b = torch_a.relu().sum(dim=None, keepdim=True)
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # relu, sum  # TODO replace with leaky_relu
    kitty_b = kitty_a.relu().sum(dim=())
    torch_b = torch_a.relu()  # TODO: ref [2]
    assert compare(kitty_b, torch_b)

    kitty_b.backward(kitty.ones_like(kitty_b))
    torch_b.backward(torch.ones_like(torch_b))
    assert compare(kitty_a.grad, torch_a.grad)

    # TODO: SiLU


def test_activation_exceptions():
    pass


def test_inplace():
    pass


def test_inplace_exceptions():
    pass


@pytest.mark.parametrize(
    'shape', [
        (4, 1, 3, 2, 1, 5),
        (1, 2, 3, 4, 1, 6),
    ])
def test_view(shape, compare):
    print()
    kitty_a, torch_a = map(next, init_tensors(shape))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    def test():
        assert compare(kitty_b, torch_b)

        kitty_b.sum().backward()
        torch_b.sum().backward()
        assert compare(kitty_a.grad, torch_a.grad)

        zero_grad()

    # transpose
    kitty_b = 1 / (2 * kitty_a).transpose(-3, 2)
    torch_b = 1 / (2 * torch_a).transpose(-3, 2)
    test()

    # transpose the same dim
    kitty_b = 1 / (2 * kitty_a).transpose(1, -5)
    torch_b = 1 / (2 * torch_a).transpose(1, -5)
    test()

    # mT
    kitty_b = (kitty_a + 3).mT ** -1
    torch_b = (torch_a + 3).mT ** -1
    test()

    # permute
    kitty_b = -kitty_a.relu().permute((-1, 0, 3, -2, 1, -4))
    torch_b = -torch_a.relu().permute((-1, 0, 3, -2, 1, -4))
    test()

    # squeeze dim=None
    kitty_b = kitty_a.squeeze()
    torch_b = torch_a.squeeze()
    test()

    # squeeze dim=tuple
    kitty_b = kitty_a.squeeze((1, -6))
    torch_b = torch_a.squeeze((1, -6))
    test()

    # squeeze dim=list
    kitty_b = kitty_a.squeeze([-1, -2])
    torch_b = torch_a.squeeze([-1, -2])
    test()

    # squeeze dim=int
    kitty_b = kitty_a.squeeze(-2)
    torch_b = torch_a.squeeze(-2)
    test()

    # unsqueeze dim=tuple
    kitty_b = kitty_a.unsqueeze((0, 7))
    torch_b = torch_a.unsqueeze(0).unsqueeze(7)
    test()

    # unsqueeze dim=list
    kitty_b = kitty_a.unsqueeze([-1, -2])
    torch_b = torch_a.unsqueeze(-1).unsqueeze(-1)
    test()

    # unsqueeze dim=int
    kitty_b = kitty_a.unsqueeze(4)
    torch_b = torch_a.unsqueeze(4)
    test()

    # expand
    kitty_b = kitty_a.expand(13, -1, -1, 3, -1, 6, -1)
    torch_b = torch_a.expand(13, -1, -1, 3, -1, 6, -1)
    test()

    # expand without any impact
    kitty_b = kitty_a.expand(-1, -1, -1, -1, -1, -1)
    torch_b = torch_a.expand(-1, -1, -1, -1, -1, -1)
    assert kitty_b is kitty_b

    kitty_b.sum().backward()
    torch_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)


def test_view_exceptions():
    pass


@pytest.mark.parametrize(
    'shape_a,shape_b,shape_c', [
        ((10, 1, 2), (3, 1), (2, 1, 1, 1)),
    ])
def test_broadcast_tensors(shape_a, shape_b, shape_c, compare):
    print()
    init_kitty, init_torch = init_tensors(shape_a, shape_b, shape_c)
    kitty_a, kitty_b, kitty_c = init_kitty
    torch_a, torch_b, torch_c = init_torch

    kitty_b.requires_grad = False
    torch_b.requires_grad = False

    kitty_a_view, kitty_b_view, kitty_c_view = kitty.broadcast_tensors(kitty_a, kitty_b, kitty_c)
    torch_a_view, torch_b_view, torch_c_view = torch.broadcast_tensors(torch_a, torch_b, torch_c)

    assert compare(kitty_a_view, torch_a_view)
    assert compare(kitty_b_view, torch_b_view)
    assert compare(kitty_c_view, torch_c_view)

    kitty_a_view.sum().backward()
    torch_a_view.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    with pytest.raises(RuntimeError) as msg:
        kitty_b_view.sum().backward()
    assert str(msg.value) == "Tensor does not require grad and does not have a grad_fn."
    assert kitty_b.grad is None

    kitty_c_view.sum().backward()
    torch_c_view.sum().backward()
    assert compare(kitty_c.grad, torch_c.grad)


@pytest.mark.parametrize(
    'shape_a,shape_b,shape_c,shape_d', [
        ((1,), (1,), (1,), (1,)),
    ])
def test_engine(shape_a, shape_b, shape_c, shape_d, compare):
    print()
    init_kitty, init_torch = init_tensors(shape_a, shape_b, shape_c, shape_d)
    kitty_a, kitty_b, kitty_c, kitty_d = init_kitty
    torch_a, torch_b, torch_c, torch_d = init_torch

    def zero_grad():
        for tensor in (kitty_a, kitty_b, kitty_c, kitty_d,
                       torch_a, torch_b, torch_c, torch_d):
            tensor.grad = None

    # incomplete backpropagation warning
    kitty_b.requires_grad = False
    kitty_e = kitty_a * kitty_b
    kitty_f = kitty_e + 1
    kitty_f.retain_grad()
    kitty_g = (kitty_f.tanh() * 8 + 2) * 0.5
    kitty_h = 2 * kitty_f + 3
    kitty_i = kitty_e * 9

    for kitty_t in (kitty_g, kitty_h):
        with pytest.warns(UserWarning, match="Backpropagation not completed"):
            kitty_t.sum().backward()

    kitty_i.sum().backward()

    torch_b.requires_grad = False
    torch_e = torch_a * torch_b
    torch_f = torch_e + 1
    torch_f.retain_grad()
    torch_g = (torch_f.tanh() * 8 + 2) * 0.5
    torch_h = 2 * torch_f + 3
    torch_i = torch_e * 9

    (torch_g.sum() + torch_h.sum() + torch_i.sum()).backward()

    assert kitty_b.grad is None
    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_f.grad, torch_f.grad)

    kitty_b.requires_grad = True
    torch_b.requires_grad = True
    zero_grad()

    # separate graphs with the same leaves
    kitty_e = kitty_a * kitty_b * kitty_c * kitty_d
    kitty_f = kitty_a + kitty_b + kitty_c + kitty_d
    kitty_e.sum().backward()
    kitty_f.sum().backward()

    torch_e = torch_a * torch_b * torch_c * torch_d
    torch_f = torch_a + torch_b + torch_c + torch_d
    torch_e.sum().backward()
    torch_f.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)
    assert compare(kitty_d.grad, torch_d.grad)

    zero_grad()

    # redundant .backward() call
    kitty_e = (kitty_a + kitty_b).sum()
    kitty_e.backward()
    with pytest.raises(RuntimeError) as msg:
        kitty_e.backward()
    assert str(msg.value) == "Trying to backward through the graph a second time."

    zero_grad()

    # calling .backward() from the middle of the graph plus the previous case
    kitty_e = kitty_a + kitty_b
    kitty_f = kitty_e * kitty_c
    kitty_g = kitty_f + kitty_d

    with pytest.warns(UserWarning,
                      match=r"A .backward\(\) call from the middle of the computational graph was noticed."):
        kitty_f.backward(kitty.ones_like(kitty_f))

    torch_e = torch_a + torch_b
    torch_f = torch_e * torch_c
    torch_f.backward(torch.ones_like(torch_f))

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)

    with pytest.raises(RuntimeError) as msg:
        kitty_g.backward(kitty.ones_like(kitty_g))
    assert str(msg.value) == "Trying to backward through the graph a second time."

    zero_grad()

    # redundant .backward() call on the last temporal tensor
    kitty_e = kitty_a * kitty_b
    kitty_f = kitty_e + 1
    kitty_g = kitty_e + (-1)
    (kitty_f + kitty_g).sum().backward()
    with pytest.raises(RuntimeError) as msg:
        (kitty_f + kitty_g).sum().backward()
    assert str(msg.value) == "Trying to backward through the graph a second time."

    zero_grad()

    # calling .backward() from the middle of the graph plus incomplete backprop
    kitty_e = kitty_a + kitty_b
    kitty_f = kitty_e + 2
    kitty_g = 2 * kitty_e
    _ = kitty_f * kitty_f

    with pytest.warns(Warning) as warn_info:
        kitty_f.backward(kitty.ones_like(kitty_f))
    warns = set((warn.category, warn.message.args[0]) for warn in warn_info)
    expected = {
        (UserWarning, "A .backward() call from the middle of the computational graph was noticed."),
        (UserWarning, "Backpropagation not completed. The computational graph "
                      "has at least one more output for the .backward() call.")
    }
    assert warns == expected

    kitty_g.sum().backward()

    torch_e = torch_a + torch_b
    torch_f = torch_e + 2
    torch_g = 2 * torch_e
    (torch_f + torch_g).sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)

    zero_grad()

    # gradient accumulation without a computational graph
    kitty_a.backward(kitty.ones_like(kitty_a))
    torch_a.backward(torch.ones_like(torch_a))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # gradient accumulation without a computational graph with wrong dtype
    with pytest.raises(TypeError) as msg:
        kitty_a.backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    assert str(msg.value) == "Assigned grad has data of a different type."

    zero_grad()

    # the engine fixes wrong initial gradient dtype TODO
    (kitty_a * 2).backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    (torch_a * 2).backward(torch.ones_like(torch_a, dtype=torch.float16))
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()


def test_more():
    pass
