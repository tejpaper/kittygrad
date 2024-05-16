from __future__ import annotations

import kittygrad.core as core
import kittygrad.tensor.tensor as tsr
from kittygrad.autograd.activation import (
    SigmoidBackward,
    TanhBackward,
    ReluBackward,
    SoftmaxBackward,
)
from kittygrad.autograd.engine import BackwardGraph
from kittygrad.func.utils import dim2tuple, check_dims


@BackwardGraph.mount(SigmoidBackward)
def _sigmoid(_ctx: Context, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=1 / (1 + core.strict.exp(-tensor._data)),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(TanhBackward)
def _tanh(_ctx: Context, tensor: Tensor) -> Tensor:
    exp = core.strict.exp(2 * tensor._data)
    return tsr.tensor(
        data=(exp - 1) / (exp + 1),  # TODO: try np.tanh
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(ReluBackward)
def _relu(_ctx: Context, tensor: Tensor) -> Tensor:
    return tsr.tensor(
        data=tensor._data * (tensor._data > 0),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


@BackwardGraph.mount(SoftmaxBackward)
def _softmax(ctx: Context, tensor: Tensor, dim: int | Size | None) -> Tensor:  # TODO: test
    dim = dim2tuple(dim, tensor.ndim)
    check_dims(dim, tensor.ndim)

    exp = core.strict.exp(tensor._data - core.np.max(tensor._data))
    probs = core.strict.divide(exp, core.np.sum(exp, axis=dim, keepdims=True))

    ctx.dim = dim
    return tsr.tensor(
        data=probs,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    )


def softmax():
    pass  # TODO: frontend softmax function


def _log_softmax():
    pass  # TODO


def log_softmax():
    pass  # TODO
