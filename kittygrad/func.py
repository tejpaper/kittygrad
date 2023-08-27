from __future__ import annotations

import kittygrad.tensor as tsr
from .utils import *
from .autograd import (
    AccumulateGrad,
    AddBackward,
    MulBackward,
)


def add(tensor: tsr.Tensor, other: Scalar | tsr.Tensor) -> tsr.Tensor:
    return tensor  # TODO


def mul(tensor: tsr.Tensor, other: Scalar | tsr.Tensor) -> tsr.Tensor:
    return tensor  # TODO
