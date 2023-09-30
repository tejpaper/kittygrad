from .tensor import (
    # tensor.py
    tensor, Tensor,
    # create.py
    rand, randn, ones, ones_like, zeros, zeros_like,
)
from .func import (
    # ops.py
    dot, mm, mv, bmm, matmul,
    # view.py
    broadcast_to, broadcast_tensors,
)
from .utils import (
    # constants.py
    Size, Scalar,
    float16, half,
    float32, float,
    float64, double,
    # functions.py
    manual_seed,
)
from .viz.dot import CompGraph

# configurations
import warnings as _warnings
_warnings.simplefilter('always', UserWarning)
