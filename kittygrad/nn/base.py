from __future__ import annotations

import abc
import itertools
import typing

import kittygrad as kitty
import kittygrad.core as core


class Parameter(kitty.Tensor):
    def __init__(self, tensor: kitty.Tensor) -> None:
        super().__init__(tensor._data, requires_grad=tensor.requires_grad)

        for attr_name, attr_value in self.__dict__.items():
            if attr_value is not (t_attr_value := getattr(tensor, attr_name)):
                setattr(self, attr_name, t_attr_value)

    def __repr__(self) -> str:
        return f'Parameter containing:\n{super().__repr__()}'


class DummyParameter(core.DotDict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.shape is None:
            self.shape = tuple()

        if self.dtype is None:
            self.dtype = core.DEFAULT_DTYPE

        if self.requires_grad is None:
            self.requires_grad = True


class Module(abc.ABC):
    _INIT_ATTRS = ('training', '_parameters', '_modules')

    def __init__(self) -> None:
        self.training = True

        self._parameters = {}
        self._modules = {}

    def __setattr__(self, attr_name: str, attr_value: typing.Any) -> None:
        if all(hasattr(self, attr) for attr in Module._INIT_ATTRS):
            if isinstance(attr_value, Parameter):
                self._parameters[attr_name] = attr_value
            elif isinstance(attr_value, Module):
                self._modules[attr_name] = attr_value

        elif isinstance(attr_value, Parameter | Module):
            raise AttributeError(f"Cannot assign {type(attr_value).__name__} instance "
                                 "before Module.__init__() call.")

        super(Module, self).__setattr__(attr_name, attr_value)

    def __delattr__(self, attr_name: str) -> None:
        super(Module, self).__delattr__(attr_name)

        if attr_name in self._parameters:
            self._parameters.pop(attr_name)
        elif attr_name in self._modules:
            self._modules.pop(attr_name)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> typing.Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> typing.Any:
        return self.forward(*args, **kwargs)

    # TODO: extra_repr

    def __repr__(self) -> str:  # TODO: test
        prefix = f'{type(self).__name__}('

        content = ''
        indent = f'\n{core.SUBMODULE_INDENT}'
        for name, submodule in self._modules.items():
            submodule_repr = repr(submodule).split('\n')
            submodule_repr[0] = f'{indent}({name}): {submodule_repr[0]}'
            content += indent.join(submodule_repr)

        return prefix + content + ('\n)' if content else ')')

    def __str__(self) -> str:
        return repr(self)

    def train(self, mode: bool = True) -> typing.Self:
        self.training = mode
        return self

    def eval(self) -> typing.Self:
        return self.train(False)

    def named_parameters(self,
                         prefix: str = '',
                         recurse: bool = True,
                         remove_duplicate: bool = True,
                         ) -> typing.Iterator[tuple[str, Parameter]]:
        params = itertools.chain(
            *map(lambda x: [(prefix + x[0], x[1])], self._parameters.items()),
            *map(lambda x: x[1].named_parameters(x[0] + core.SUBMODULE_SEPARATOR, remove_duplicate=False),
                 self._modules.items()) if recurse else []
        )
        if remove_duplicate:
            return iter({id(param): (name, param) for (name, param) in params}.values())
        else:
            return params

    def parameters(self, recurse: bool = True) -> typing.Iterator[Parameter]:
        return iter(map(lambda x: x[1], self.named_parameters(recurse=recurse)))

    def n_parameters(self) -> int:
        return sum(map(Parameter.nelement, self.parameters()))

    def n_trainable_parameters(self) -> int:
        return sum(map(lambda p: p.nelement() if p.requires_grad else 0, self.parameters()))

    def apply(self, fn: typing.Callable) -> typing.Self:
        fn(self)
        list(map(fn, self._modules.values()))
        return self

    def requires_grad_(self, requires_grad: bool = True) -> typing.Self:
        for param in self.parameters():
            param.requires_grad = requires_grad
        return self

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None
