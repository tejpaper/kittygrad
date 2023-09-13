from __future__ import annotations

from ..autograd.activation import (
    SigmoidBackward,
    TanhBackward,
    ReluBackward,
)
from ..autograd.engine import backward_graph
from ..utils import *


@backward_graph(SigmoidBackward)
def _sigmoid(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=1 / (1 + np.exp(-tensor._data)),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(TanhBackward)
def _tanh(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    exp = np.exp(2 * tensor._data)
    return tsr.tensor(
        data=(exp - 1) / (exp + 1),
        requires_grad=tensor.requires_grad,
    )


@backward_graph(ReluBackward)
def _relu(tensor: tsr.Tensor, _ctx: DotDict[str, list]) -> tsr.Tensor:
    return tsr.tensor(
        data=tensor._data * (tensor._data > 0),
        requires_grad=tensor.requires_grad,
    )


# TODO: SiLU, LeakyReLU
