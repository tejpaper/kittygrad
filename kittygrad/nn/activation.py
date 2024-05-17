import kittygrad as kitty
from kittygrad.core import Size
from kittygrad.func.activation import (
    _sigmoid,
    _tanh,
    _relu,
    _softmax,
)
from kittygrad.nn.base import Module


class Sigmoid(Module):
    def forward(self, input: kitty.Tensor) -> kitty.Tensor:
        return _sigmoid(input)


class Tanh(Module):
    def forward(self, input: kitty.Tensor) -> kitty.Tensor:
        return _tanh(input)


class ReLU(Module):
    def forward(self, input: kitty.Tensor) -> kitty.Tensor:
        return _relu(input)


class Softmax(Module):
    def __init__(self, dim: int | Size | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: kitty.Tensor) -> kitty.Tensor:
        return _softmax(input, self.dim)
