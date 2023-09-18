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
    # TODO: test autotype


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

    with pytest.raises(NotImplementedError) as msg:
        np.array(2, dtype=np.int8) + kitty_a
    assert str(msg.value) == "Unsupported operation with NumPy array. Try swapping operands."
    kitty_a + np.array(2, dtype=np.int8)  # it's ok, auto cast

    with pytest.raises(TypeError) as msg:
        kitty_a.type(np.int8)
    assert str(msg.value) == "Data type 'int8' is not supported."


@pytest.mark.parametrize(
    'shape', [
        (2, 3, 4, 5),
        (2, 1, 8, 1),
    ])
def test_agg(shape, compare):
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


@pytest.mark.parametrize(
    'shape', [
        (2, 3, 4, 5),
        (2, 1, 8, 1),
    ])
def test_agg_exceptions(shape):
    kitty_a, _ = map(next, init_tensors(shape))

    with pytest.raises(RuntimeError) as msg:
        kitty_a.sum((0, 0))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(RuntimeError) as msg:
        kitty_a.mean((0, -4))
    assert str(msg.value) == "Duplicate dims are not allowed."

    with pytest.raises(IndexError) as msg:
        kitty_a.sum((42, 0))  # TODO: replace with std
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got 42)."

    with pytest.raises(IndexError) as msg:
        kitty_a.sum(-5)
    assert str(msg.value) == "Dimension out of range (expected to be in range of [-4, 3], but got -5)."


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
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shape_a, shape_b)

    kitty_c = kitty_a @ kitty_b
    torch_c = torch_a @ torch_b
    assert compare(kitty_c, torch_c)

    kitty_c.sum().backward()
    torch_c.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)


def test_matmul_exceptions():
    # mm
    (kitty_a, kitty_b), _ = init_tensors((2, 3), (3, 4))
    with pytest.raises(TypeError) as msg:
        kitty.mm(kitty_a.type(kitty.float16), kitty_b)
    assert str(msg.value) == "Operands type mismatch: float16 != float32."

    (kitty_a, kitty_b), _ = init_tensors((3,), (3, 4))
    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "2D tensors expected, but got 1D and 2D tensors."

    (kitty_a, kitty_b), _ = init_tensors((2, 3), (3, 4, 1, 1))
    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "2D tensors expected, but got 2D and 4D tensors."

    (kitty_a, kitty_b), _ = init_tensors((2, 3), (4, 3))
    with pytest.raises(RuntimeError) as msg:
        kitty.mm(kitty_a, kitty_b)
    assert str(msg.value) == "input and mat2 shapes cannot be multiplied (2x3 and 4x3)."

    # dot
    (kitty_a, kitty_b), _ = init_tensors((16,), (16,))
    with pytest.raises(TypeError) as msg:
        kitty.dot(kitty_a.type(kitty.float16), kitty_b)
    assert str(msg.value) == "Operands type mismatch: float16 != float32."

    (kitty_a, kitty_b), _ = init_tensors((16, 16), (16,))
    with pytest.raises(RuntimeError) as msg:
        kitty.dot(kitty_a, kitty_b)
    assert str(msg.value) == "1D tensors expected, but got 2D and 1D tensors."

    (kitty_a, kitty_b), _ = init_tensors((18,), (16,))
    with pytest.raises(RuntimeError) as msg:
        kitty.dot(kitty_a, kitty_b)
    assert str(msg.value) == ("Inconsistent tensor size, expected tensor input and other to have "
                              "the same number of elements, but got 18 and 16 elements respectively.")

    # mv
    (kitty_a, kitty_b), _ = init_tensors((3, 4), (4,))
    with pytest.raises(TypeError) as msg:
        kitty.mv(kitty_a, kitty_b.type(kitty.float64))
    assert str(msg.value) == "Operands type mismatch: float32 != float64."

    (kitty_a, kitty_b), _ = init_tensors((2, 3, 4), (4,))
    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "input must be a matrix, not a 3D tensor."

    (kitty_a, kitty_b), _ = init_tensors((3, 4), (4, 1))
    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "vec must be a vector, not a 2D tensor."

    (kitty_a, kitty_b), _ = init_tensors((3, 4), (5,))
    with pytest.raises(RuntimeError) as msg:
        kitty.mv(kitty_a, kitty_b)
    assert str(msg.value) == "input and vec shapes cannot be multiplied (3x4 and 5)."

    # bmm
    (kitty_a, kitty_b), _ = init_tensors((2, 3, 4), (2, 4, 2))
    with pytest.raises(TypeError) as msg:
        kitty.bmm(kitty_a, kitty_b.type(kitty.float64))
    assert str(msg.value) == "Operands type mismatch: float32 != float64."

    (kitty_a, kitty_b), _ = init_tensors((2, 3, 4), (1, 2, 4, 2))
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("Batch dimensions of both tensors must be equal, but got "
                              "(2,) and (1, 2) respectively.")

    (kitty_a, kitty_b), _ = init_tensors((2, 3, 4), (4, 2))
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("Batch dimensions of both tensors must be equal, but got "
                              "(2,) and () respectively.")

    (kitty_a, kitty_b), _ = init_tensors((4,), (4, 2))
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == ("The batch matrix-matrix product requires the "
                              "tensors to have at least 3 dimensions each.")

    (kitty_a, kitty_b), _ = init_tensors((13, 2, 3, 5), (13, 2, 4, 2))
    with pytest.raises(RuntimeError) as msg:
        kitty.bmm(kitty_a, kitty_b)
    assert str(msg.value) == "input and mat2 matrix shapes cannot be multiplied (3x5 and 4x2)."

    # matmul
    with pytest.raises(RuntimeError) as msg:
        kitty.matmul(kitty.tensor(1), kitty.tensor([2, 3]))
    assert str(msg.value) == "Input tensors must not be scalars."


@pytest.mark.parametrize(
    'shape_a,shape_b,squeeze_dims', [
        ((2, 3), (2, 3), ()),
        ((1,), (1,), ()),
        ((1,), (1,), (0,)),
        ((5, 3, 4, 1), (3, 1, 1), ()),
    ])
def test_inplace(shape_a, shape_b, squeeze_dims, compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors(shape_a, shape_b, squeeze_dims=squeeze_dims)

    def zero_grad():
        for tensor in (kitty_a, kitty_b, torch_a, torch_b):
            tensor.grad = None

    # __iadd__
    kitty_c = kitty_a + 0
    kitty_c += kitty_b
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a + 0
    torch_c += torch_b
    torch_d = torch_c * torch_b

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)

    kitty_c.retain_grad()
    kitty_d.sum().backward()
    torch_c.retain_grad()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)

    zero_grad()

    # __isub__
    kitty_c = kitty_a.type(kitty.float64)
    kitty_c -= kitty_b
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a.type(torch.float64)
    torch_c -= torch_b
    torch_d = torch_c * torch_b

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)

    kitty_c.retain_grad()
    kitty_d.sum().backward()
    torch_c.retain_grad()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)

    zero_grad()

    # __imul__
    kitty_c = kitty_a + 0
    kitty_c *= kitty_b
    kitty_d = 1 / kitty_c

    torch_c = torch_a + 0
    torch_c *= torch_b
    torch_d = 1 / torch_c

    assert compare(kitty_a, torch_a)
    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)

    kitty_c.retain_grad()
    kitty_d.sum().backward()
    torch_c.retain_grad()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)

    zero_grad()

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

    # __ipow__
    kitty_c = kitty_a + 0
    kitty_c **= 2
    kitty_c **= 0.5
    kitty_c **= kitty_b
    kitty_d = kitty_c * kitty_b

    torch_c = torch_a + 0
    torch_c **= 2
    torch_c **= 0.5
    torch_c **= torch_b
    torch_d = torch_c * torch_b

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)

    kitty_c.retain_grad()
    kitty_d.sum().backward()
    torch_c.retain_grad()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)
    assert compare(kitty_c.grad, torch_c.grad)


def test_inplace_exceptions(compare):
    kitty_a = kitty.tensor([[1]])

    with pytest.raises(NotImplementedError) as msg:
        kitty_a @= kitty.tensor([[2]])
    assert str(msg.value) == "Inplace matrix multiplication is not implemented."

    with pytest.raises(TypeError) as msg:
        kitty_a **= 'string'
    assert str(msg.value) == "Unsupported operand type(s) for **=: 'Tensor' and 'str'."

    kitty_a.requires_grad = True

    with pytest.raises(RuntimeError) as msg:
        kitty_a /= 2
    assert str(msg.value) == "A leaf Variable that requires grad is being used in an in-place operation."

    kitty_b = kitty_a.expand(2, 3)

    with pytest.raises(RuntimeError) as msg:
        kitty_b += 1
    assert str(msg.value) == ("The inplace operation cannot be applied to a read-only tensor. If this "
                              "tensor is a view of another, you can try to do the same operation with it.")

    kitty_a = kitty.tensor([[2], [2]], dtype=kitty.float16)
    kitty_b = kitty.tensor([[2], [2]], requires_grad=True)

    with pytest.raises(RuntimeError) as msg:
        kitty_a *= kitty_b
    assert str(msg.value) == "Output with dtype 'float16' doesn't match the broadcast dtype 'float32'."

    kitty_a = kitty.tensor([3, 3, 3])

    with pytest.raises(RuntimeError) as msg:
        kitty_a **= kitty_b
    assert str(msg.value) == "Output with shape (3,) doesn't match the broadcast shape (2, 3)."

    # converting a leaf tensor to a non-leaf tensor using inplace operation
    kitty_a = kitty.tensor(2., requires_grad=False)
    kitty_b = kitty.tensor(3., requires_grad=True)
    kitty_a *= kitty_b
    kitty_a.backward()

    torch_a = torch.tensor(2., requires_grad=False)
    torch_b = torch.tensor(3., requires_grad=True)
    torch_a *= torch_b
    torch_a.backward()

    assert compare(kitty_a, torch_a)
    assert compare(kitty_b.grad, torch_b.grad)

    with pytest.warns(UserWarning,
                      match="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed."):
        kitty_a.grad
