from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    SigmoidBackward,
    TanhBackward,
    ReluBackward,
)

import numpy as np


# noinspection PyProtectedMember
@backward_graph(SigmoidBackward)
def sigmoid(tensor: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=1 / (1 + np.exp(-tensor._data)),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    ),


# noinspection PyProtectedMember
@backward_graph(TanhBackward)
def tanh(tensor: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    exp = np.exp(2 * tensor._data)
    return tsr.tensor(
        data=(exp - 1) / (exp + 1),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    ),


# noinspection PyProtectedMember
@backward_graph(ReluBackward)
def relu(tensor: tsr.Tensor) -> tuple[tsr.Tensor, ...]:
    return tsr.tensor(
        data=tensor._data * (tensor._data > 0),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    ),
