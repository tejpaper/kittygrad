__version__ = '1.0.0-alpha'

from kittygrad.autograd.interaction import no_grad, Function

from kittygrad.func.activation import softmax, log_softmax
from kittygrad.func.ops import mm, dot, mv, bmm, matmul
from kittygrad.func.view import broadcast_to, broadcast_tensors

from kittygrad.core import (
    Size, Scalar,
    float16, half,
    float32, float,
    float64, double,
)

from kittygrad.tensor.create import (
    rand, randn,
    ones, ones_like,
    zeros, zeros_like,
    empty, empty_like,
)
from kittygrad.tensor.tensor import Tensor, tensor

from kittygrad.viz.dot import CompGraph

import warnings
import kittygrad.core as core

warnings.simplefilter('always', UserWarning)
manual_seed = core.np.random.seed

__all__ = core.prepare_namespace(module_name=__name__) + [
    'Size', 'Scalar',
    'float16', 'half',
    'float32',
    'float64', 'double',
    'manual_seed',
]

del core, warnings
