from test.conftest import *


def test_no_grad(compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors([(2, 3), (2, 3)])

    def zero_grad():
        for tensor in (kitty_a, kitty_b, torch_a, torch_b):
            tensor.grad = None

    # trivial context manager test
    with kitty.no_grad():
        kitty_c = kitty_a * kitty_b
    assert not kitty_c.requires_grad
    assert kitty_c.grad_fn is None

    # trivial decorator test
    @kitty.no_grad
    @torch.no_grad()
    def sub(t1, t2):
        return t1 - t2

    kitty_c = sub(kitty_a, kitty_b)
    assert not kitty_c.requires_grad
    assert kitty_c.grad_fn is None

    # no_grad has no effect on future backward passes
    with kitty.no_grad():
        kitty_c = kitty_a * kitty_b
    kitty_d = kitty_c + kitty_a

    with torch.no_grad():
        torch_c = torch_a * torch_b
    torch_d = torch_c + torch_a

    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    torch_d.sum().backward()

    assert kitty_b.grad is None
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # no_grad plus inplace
    with kitty.no_grad(), torch.no_grad():
        kitty_a += 2
        torch_a += 2

    assert compare(kitty_a, torch_a)

    # nested context managers
    with kitty.no_grad():
        kitty_c = kitty_a + kitty_b
        with pytest.warns(UserWarning, match="Calling no_grad more then once has no additional effect."):
            with kitty.no_grad():
                kitty_d = kitty_c + 1
        kitty_e = kitty_d + kitty_a
    with kitty.no_grad():
        kitty_f = kitty_a + kitty_b

    assert not kitty_c.requires_grad
    assert not kitty_d.requires_grad
    assert not kitty_e.requires_grad
    assert not kitty_f.requires_grad

    # two decorated functions
    @kitty.no_grad
    @torch.no_grad()
    def mul(t1, t2):
        return t1 * t2

    kitty_c = sub(kitty_a, kitty_b)
    kitty_d = kitty_c * kitty_a
    kitty_e = mul(kitty_d, kitty_b)

    torch_c = sub(torch_a, torch_b)
    torch_d = torch_c * torch_a
    torch_e = mul(torch_d, torch_b)

    assert compare(kitty_c, torch_c)
    assert compare(kitty_d, torch_d)
    assert compare(kitty_e, torch_e)

    kitty_d.sum().backward()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert kitty_b.grad is None

    zero_grad()

    # nested decorators
    @kitty.no_grad
    @torch.no_grad()
    def nested_transformation_1(t1, t2):
        t3 = t1 * t2

        @kitty.no_grad
        @torch.no_grad()
        def inv_mul(t):
            return t ** -1

        return inv_mul(t3)

    # context manager nested in decorator
    @kitty.no_grad
    @torch.no_grad()
    def nested_transformation_2(t1, t2):
        t3 = t1 * t2

        with kitty.no_grad(), torch.no_grad():
            t4 = t3 - t1

        return t4.abs()

    for nested_transformation in (nested_transformation_1, nested_transformation_2):
        kitty_c = kitty_a * 2
        with pytest.warns(UserWarning, match="Calling no_grad more then once has no additional effect."):
            kitty_d = nested_transformation(kitty_c, kitty_b)
        kitty_e = kitty_d * kitty_c

        torch_c = torch_a * 2
        with pytest.warns(UserWarning, match="Calling no_grad more then once has no additional effect."):
            torch_d = nested_transformation(torch_c, torch_b)
        torch_e = torch_d * torch_c

        assert compare(kitty_c, torch_c)
        assert compare(kitty_d, torch_d)
        assert compare(kitty_e, torch_e)

        kitty_e.sum().backward()
        torch_e.sum().backward()

        assert compare(kitty_a.grad, torch_a.grad)
        assert kitty_b.grad is None

        zero_grad()

    # decorator nested within context manager
    with pytest.warns(UserWarning, match="Calling no_grad more then once has no additional effect."):
        with kitty.no_grad():
            kitty_c = sub(kitty_a, kitty_b)

    assert not kitty_c.requires_grad
    assert kitty_c.grad_fn is None
