from __future__ import annotations

import numpy as np

import kittygrad.tensor.tensor as tsr
from kittygrad.autograd.activation import (
    SigmoidBackward,
    TanhBackward,
    ReluBackward,
)
from kittygrad.autograd.engine import BackwardGraph
from kittygrad.utils.classes import DotDict


@BackwardGraph.mount(SigmoidBackward)
def _sigmoid(_ctx: DotDict, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=1 / (1 + np.exp(-tensor._data)),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(TanhBackward)
def _tanh(_ctx: DotDict, tensor: Tensor) -> Tensor:
    exp = np.exp(2 * tensor._data)
    return tsr.tensor(
        data=(exp - 1) / (exp + 1),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(ReluBackward)
def _relu(_ctx: DotDict, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=tensor._data * (tensor._data > 0),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )
