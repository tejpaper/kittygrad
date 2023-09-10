from .tensor import (
    Tensor, tensor,
    rand,
    randn,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .func import (
    dot,
    mm,
    mv,
    bmm,
    matmul,
    broadcast_tensors,
)

from numpy import float16 as float16
from numpy import float16 as half

from numpy import float32 as float32
from numpy import float32 as float  # noqa: torch like API

from numpy import float64 as float64
from numpy import float64 as double

from numpy.random import seed as manual_seed
