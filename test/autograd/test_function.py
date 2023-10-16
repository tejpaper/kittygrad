import pytest

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
        x = self.ctx.x_array
        return kitty.tensor(np.cos(x))


class Sin2(SinTemplate, output_version_check=True):
    def backward(self, grad):
        sgn_cos = ((self.ctx.x_array - np.pi / 2) % (2 * np.pi) > np.pi) * 2 - 1
        return ((1 - self.ctx.out ** 2) ** 0.5) * sgn_cos.astype(np.float32)


def test_function(compare):
    kitty_a, torch_a = map(next, init_tensors([(2, 3)]))

    def zero_grad():
        kitty_a.grad = None
        torch_a.grad = None

    cube = Cube()
    sin_1 = Sin1()
    sin_2 = Sin2()

    # abstract class initialization
    with pytest.raises(TypeError) as msg:
        SinTemplate()
    assert str(msg.value) == "Can't instantiate abstract class SinTemplate with abstract method backward"

    # Cube backward
    kitty_b = cube(kitty_a)
    torch_b = torch_a ** 3
    assert f'{kitty_b.grad_fn}' == '<CubeBackward>'
    assert compare(kitty_b, torch_b)

    kitty_b.sum().backward()
    torch_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    # built-in backward pass
    kitty_a.grad = None

    kitty_b = cube.forward(kitty_a)
    assert compare(kitty_b, torch_b)

    kitty_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # Sin1 backward
    assert sin_1.forward(kitty_a).grad_fn is None

    kitty_b = sin_1(kitty_a)
    torch_b = torch_a.sin()
    assert f'{kitty_b.grad_fn}' == '<Sin1Backward>'
    assert compare(kitty_b, torch_b)

    kitty_b.sum().backward()
    torch_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    # Sin2 backward
    kitty_a.grad = None

    kitty_b = sin_2(kitty_a)
    assert f'{kitty_b.grad_fn}' == '<Sin2Backward>'
    assert compare(kitty_b, torch_b)

    kitty_b.sum().backward()
    assert compare(kitty_a.grad, torch_a.grad)

    zero_grad()

    # saved_tensors version control
    kitty_b = kitty_a * 2
    kitty_c = cube(kitty_b)
    kitty_c += 1
    assert len(cube.ctx.saved_tensors) == 1

    kitty_b += 1
    with pytest.raises(RuntimeError) as msg:
        kitty_c.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # output version control
    kitty_b = sin_2(kitty_a)
    kitty_b += 1
    with pytest.raises(RuntimeError) as msg:
        kitty_b.sum().backward()
    assert str(msg.value) == ("One of the variables needed for gradient computation "
                              "has been modified by an inplace operation.")

    # retain_grad compatibility
    kitty_b = cube(kitty_a * 2)
    kitty_b.retain_grad()
    kitty_b += 1

    torch_b = (torch_a * 2) ** 3
    torch_b.retain_grad()
    torch_b += 1

    assert compare(kitty_b, torch_b)

    with pytest.warns(UserWarning, match="An attempt to assign a gradient to a tensor with retains_grad=True"):
        kitty_b.sum().backward()
    torch_b.sum().backward()

    assert compare(kitty_a.grad, torch_a.grad)
    assert compare(kitty_b.grad, torch_b.grad)

    # declaring no_grad decorator in the forward method
    with pytest.raises(RuntimeError) as msg:
        class SomethingWentWrong1(kitty.Function):
            @kitty.no_grad
            def forward(self, x):
                return x

            def backward(self, grad):
                return grad
    assert str(msg.value) == ("There is no point in creating a Function class with gradient flow disabled. "
                              "Use a standard function instead.")

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
        kitty_b = cube(kitty_a)

    assert not kitty_b.requires_grad
    assert kitty_b.grad_fn is None

    # TODO: ref to CompGraph example
