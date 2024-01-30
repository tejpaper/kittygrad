from __future__ import annotations

from kittygrad.nn.base import *
from kittygrad.nn.init import kaiming_uniform


class Identity(Module):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    def forward(self, input: kitty.Tensor) -> kitty.Tensor:  # noqa: torch-like API
        return input


class Linear(Module):  # TODO: test
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: type | np.dtype | None = None,
                 ) -> None:
        super().__init__()

        self.weight = DummyParameter(shape=(in_features, out_features), dtype=dtype)
        self.bias = DummyParameter(shape=(out_features,), dtype=dtype) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = Parameter(kaiming_uniform(self.weight.shape,
                                                self.weight.dtype,
                                                self.weight.requires_grad,
                                                nonlinearity='linear'))
        if self.bias is not None:
            self.bias = Parameter(kitty.zeros_like(self.bias,
                                                   self.weight.dtype,
                                                   self.bias.requires_grad))

    def forward(self, input: kitty.Tensor) -> kitty.Tensor:  # noqa: torch-like API
        output = kitty.matmul(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output
