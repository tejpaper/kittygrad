import typing

import numpy as np

# auxiliary typing
Size = typing.Sequence[int] | tuple[int, ...]  # *size notation requires tuple
Scalar = int | float

# supported dtypes
float16 = half = np.float16
float32 = float = np.float32  # noqa: torch-like API
float64 = double = np.float64
ALL_DTYPES = [float16, float32, float64]
DEFAULT_DTYPE = float32

# "strict" namespace definition; all np.ndarray castings must be explicit
NP_STRICT_CONFIG = dict(casting='no')

# tensor representation
REPR_MAX_LINE_WIDTH = 89
REPR_PRECISION = 4
REPR_SEPARATOR = ', '
REPR_PREFIX = 'tensor('
REPR_FLOATMODE = 'maxprec_equal'

# module representation
SUBMODULE_SEPARATOR = '.'
SUBMODULE_INDENT = ' ' * 2
