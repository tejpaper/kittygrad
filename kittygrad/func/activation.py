from __future__ import annotations

import kittygrad.tensor.tensor as tsr
from kittygrad.autograd.activation import (
    SigmoidBackward,
    TanhBackward,
    ReluBackward,
    SoftmaxBackward,
)
from kittygrad.autograd.engine import BackwardGraph
from kittygrad.utils.classes import DotDict
from kittygrad.utils.functions import *


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


@BackwardGraph.mount(SoftmaxBackward)
def _softmax(ctx: DotDict, tensor: Tensor, dim: int | Size | None) -> Tensor:  # TODO: test
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    exp = np.exp(tensor._data - np.max(tensor._data))
    probs = np.divide(exp, np.sum(exp, axis=dim, keepdims=True), **NP_OPS_CONFIG)

    ctx.dim = dim
    return tsr.tensor(
        data=probs,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )
