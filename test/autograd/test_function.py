from test.conftest import *


class Cube(kitty.Function):
    def forward(self, x):
        self.ctx.saved_tensors.append(x)
        return x ** 3

    def backward(self, grad):
        return 3 * self.ctx.saved_tensors[0] ** 2 * grad


class SinTemplate(kitty.Function):
    def forward(self, x):
        self.ctx.x_array = x.numpy()
        return kitty.tensor(np.sin(self.ctx.x_array), requires_grad=x.requires_grad)


class Sin1(SinTemplate):
    def backward(self, grad):
        return kitty.tensor(np.cos(self.ctx.x_array)) * grad


class Sin2(SinTemplate, output_version_check=True):
    def backward(self, grad):
        sgn_cos = ((self.ctx.x_array - np.pi / 2) % (2 * np.pi) > np.pi) * 2 - 1
        return ((1 - self.ctx.out ** 2) ** 0.5) * sgn_cos.astype(np.float32) * grad


class Mess(kitty.Function):
    def forward(self, x, injection_type, injection_1, injection_2, *, is_neg=False):
        self.ctx.injection_type = injection_type
        self.ctx.is_neg = is_neg

        if injection_type == 1:
            self.ctx.saved_tensors.extend([x, injection_1])
            x = x * injection_1

        elif injection_type == 2:
            self.ctx.saved_tensors.extend([x, injection_2])
            x = x * injection_2

        return -x if is_neg else x

    def backward(self, grad):
        if self.ctx.is_neg:
            grad *= -1

        x_grad = grad.clone()
        injection_1_grad = None
        injection_2_grad = None

        if self.ctx.injection_type == 1:
            x_grad *= self.ctx.saved_tensors[1]
            injection_1_grad = grad * self.ctx.saved_tensors[0]

        elif self.ctx.injection_type == 2:
            x_grad *= self.ctx.saved_tensors[1]
            injection_2_grad = grad * self.ctx.saved_tensors[0]

        return x_grad, injection_1_grad, injection_2_grad


def test_function(compare):
    init_kitty, init_torch = init_tensors([(2, 3), (2, 3), (2, 3)])
    kitty_a, kitty_b, kitty_c = init_kitty
    torch_a, torch_b, torch_c = init_torch

    kitty_b.requires_grad = False
    torch_b.requires_grad = False

    def zero_grad():
        kitty_a.grad = None
        kitty_c.grad = None
        torch_a.grad = None
        torch_c.grad = None

    cube = Cube()
    sin_1 = Sin1()
    sin_2 = Sin2()
    mess = Mess()

    # abstract class initialization
    with pytest.raises(TypeError) as msg:
        SinTemplate()
    assert str(msg.value) == "Can't instantiate abstract class SinTemplate with abstract method backward"

    # Cube backward
    kitty_d = cube(kitty_a)
    torch_d = torch_a ** 3
    assert f'{kitty_d.grad_fn}' == '<CubeBackward>'
    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    torch_d.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    # built-in backward pass
    kitty_a.grad = None

    kitty_d = cube.forward(kitty_a)
    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # Sin1 backward
    assert sin_1.forward(kitty_a).grad_fn is None

    kitty_d = sin_1(kitty_a)
    torch_d = torch_a.sin()
    assert f'{kitty_d.grad_fn}' == '<Sin1Backward>'
    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    torch_d.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    # Sin2 backward
    kitty_a.grad = None

    kitty_d = sin_2(kitty_a)
    assert f'{kitty_d.grad_fn}' == '<Sin2Backward>'
    assert compare(kitty_d, torch_d)

    kitty_d.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # saved_tensors version control
    kitty_d = kitty_a * 2
    kitty_e = cube(kitty_d)
    kitty_e += 1
    assert len(cube.ctx.saved_tensors) == 1

    kitty_d += 1
    with pytest.raises(RuntimeError) as msg:
        kitty_e.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # output version control
    kitty_d = sin_2(kitty_a)
    kitty_d += 1
    with pytest.raises(RuntimeError) as msg:
        kitty_d.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # retain_grad compatibility
    kitty_d = cube(kitty_a * 2)
    kitty_d.retain_grad()
    kitty_d += 1

    torch_d = (torch_a * 2) ** 3
    torch_d.retain_grad()
    torch_d += 1

    assert compare(kitty_d, torch_d)

    with pytest.warns(UserWarning, match="An attempt to assign a gradient to a tensor with retains_grad=True"):
        kitty_d.sum().backward()
    torch_d.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_d.grad, torch_d.grad)

    zero_grad()

    # declaring no_grad decorator in the forward method
    with pytest.raises(RuntimeError) as msg:
        class SomethingWentWrong1(kitty.Function):
            @kitty.no_grad
            def forward(self, x):
                return x

            def backward(self, grad):
                return grad
    assert str(msg.value) == ("There is no point in creating a Function class with gradient "
                              "flow permanently disabled. Use a standard function instead.")

    # declaring no_grad decorator in the backward method
    with pytest.warns(UserWarning,
                      match=("There is no need to explicitly disable gradient flow in the backward method. "
                             "This happens implicitly.")):
        class SomethingWentWrong2(kitty.Function):
            def forward(self, x):
                return x

            @kitty.no_grad
            def backward(self, grad):
                return grad

    # no_grad context manager compatibility
    with kitty.no_grad():
        kitty_d = cube(kitty_a)

    assert not kitty_d.requires_grad
    assert kitty_d.grad_fn is None

    # multiple arguments
    kitty_d = mess(kitty_a, 1, kitty_b, kitty_c, is_neg=True)
    kitty_d.sum().backward()

    (-(torch_a * torch_b)).sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert kitty_b.grad is None and kitty_c.grad is None

    zero_grad()

    # kwargs input
    kitty_d = mess(kitty_a, injection_type=2, injection_1=kitty_b, injection_2=kitty_c)
    kitty_d.sum().backward()

    (torch_a * torch_c).sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert kitty_b.grad is None
    assert compare(kitty_c.grad, torch_c.grad)

    # missing a required argument
    with pytest.raises(TypeError) as msg:
        mess(kitty_a, injection_type=1)
    assert str(msg.value) == "missing a required argument: 'injection_1'"

    # TODO: ref to CompGraph example
