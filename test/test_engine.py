from conftest import *


@pytest.mark.parametrize(
    'shape_a,shape_b,shape_c,shape_d,squeeze_dims', [
        ((1,), (1,), (1,), (1,), ()),
        ((1,), (1,), (1,), (1,), (0,)),
    ])
def test_engine(shape_a, shape_b, shape_c, shape_d, squeeze_dims, compare):
    print()
    init_kitty, init_torch = init_tensors(shape_a, shape_b, shape_c, shape_d, squeeze_dims=squeeze_dims)
    kitty_a, kitty_b, kitty_c, kitty_d = init_kitty
    torch_a, torch_b, torch_c, torch_d = init_torch

    def zero_grad():
        for tensor in (kitty_a, kitty_b, kitty_c, kitty_d, torch_a, torch_b, torch_c, torch_d):
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

    # wrong initial gradient dtype
    with pytest.raises(TypeError) as msg:
        (kitty_a * 2).backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    assert str(msg.value) == "Assigned grad has data of a different type."

    zero_grad()

    # gradient accumulation without a computational graph with wrong dtype
    with pytest.raises(TypeError) as msg:
        kitty_a.backward(kitty.ones_like(kitty_a, dtype=kitty.float16))
    assert str(msg.value) == "Assigned grad has data of a different type."

    zero_grad()


# @pytest.mark.parametrize(
#     'shape', [
#         (1,)
#     ])
# def test_entry_point(shape):
#     print()
#     kitty_a, torch_a = map(next, init_tensors(shape))
#
#     pass

