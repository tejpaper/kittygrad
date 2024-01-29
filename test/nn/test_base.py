import kittygrad.nn as nn
from test.conftest import *


def test_parameter(compare):
    (kitty_a, kitty_b), (torch_a, torch_b) = init_tensors([(2, 3), (5, 4)])
    assert repr(nn.Parameter(kitty_a)) == repr(torch.nn.Parameter(torch_a))
    assert repr(nn.Parameter(kitty_b)) == repr(torch.nn.Parameter(torch_b))


class ModuleTest(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = nn.Parameter(weight)
        self.weight.requires_grad = True

        self.bias = nn.Parameter(bias)
        self.bias.requires_grad = False

    def forward(self, x):
        return (x * self.weight + self.bias).sum()


class DummyModule1(nn.Module):
    def __init__(self):
        self.param = nn.Parameter(kitty.empty(1, requires_grad=True))
        super().__init__()

    def forward(self, x):
        return x


class DummyModule2(nn.Module):
    def __init__(self):
        self.submodule = ModuleTest(*kitty.empty(2))
        super().__init__()

    def forward(self, x):
        return x


def test_module():
    # incorrect order of parameters initialization
    with pytest.raises(AttributeError) as msg:
        DummyModule1()
    assert str(msg.value) == "Cannot assign Parameter instance before Module.__init__() call."

    # incorrect order of submodules initialization
    with pytest.raises(AttributeError) as msg:
        DummyModule2()
    assert str(msg.value) == "Cannot assign ModuleTest instance before Module.__init__() call."

    # trivial valid example
    (kitty_a, kitty_b, kitty_c), *_ = init_tensors([(2, 2, 3), (2, 3), (5, 2, 2, 3)])
    module = ModuleTest(kitty_a, kitty_b)

    # characteristic
    assert repr(module) == str(module) == 'ModuleTest()'
    total_params_num = kitty_a.nelement() + kitty_b.nelement()
    assert module.n_parameters() == total_params_num
    assert module.n_trainable_parameters() == kitty_a.nelement()

    assert list(module.named_parameters()) == [
        ('weight', module.weight),
        ('bias', module.bias)]
    assert list(module.parameters()) == [module.weight, module.bias]

    # mode switching
    assert module.training
    module.train(False)
    assert not module.training
    module.train()
    assert module.training
    module.eval()
    assert not module.training
    module.training = True
    assert module.training

    # apply method
    nested_modules = []

    def append(m):
        nested_modules.append(m)

    module.apply(append)
    assert nested_modules == [module]

    # requires_grad_ method
    module.requires_grad_(False)
    assert module.n_trainable_parameters() == 0
    module.requires_grad_(True)
    assert module.n_trainable_parameters() == total_params_num

    # training iteration
    output = module(kitty_c)

    output.backward()
    with kitty.no_grad():
        for param in module.parameters():
            param -= 0.01 * param.grad

        assert module(kitty_c) < output

    module.zero_grad()
    assert all(param.grad is None for param in module.parameters())

    # deleting parameters
    del module.weight
    assert module.n_parameters() == kitty_b.nelement()

    # TODO: test nesting properties, leave a comment right here
