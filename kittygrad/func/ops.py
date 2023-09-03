from __future__ import annotations

import kittygrad.tensor as tsr
from ..autograd import (
    backward_graph,
    NegBackward,
    ExpBackward,
    AddBackward,
    SubBackward,
    MulBackward,
    DivBackward,
    PowBackward,
)

import numpy as np


@backward_graph(NegBackward)
def _neg(tensor: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=-tensor._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    ),


@backward_graph(ExpBackward)
def _exp(tensor: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=np.exp(tensor._data),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
    ),


@backward_graph(AddBackward)
def _add(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=tensor._data + other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ),


@backward_graph(SubBackward)
def _sub(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=tensor._data - other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ),


@backward_graph(MulBackward)
def _mul(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=tensor._data * other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ), tensor, other


@backward_graph(DivBackward)
def _div(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    other_inv = np.array(1 / other._data)
    return tsr.tensor(
        data=tensor._data * other_inv,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ), other_inv


@backward_graph(PowBackward)
def _pow(tensor: tsr.Tensor, other: tsr.Tensor) -> tuple[tsr.Tensor | np.ndarray, ...]:
    return tsr.tensor(
        data=tensor._data ** other._data,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad or other.requires_grad,
    ), tensor, other

