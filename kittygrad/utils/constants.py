import numpy as np

# auxiliary typing
Size = tuple[int, ...] | list[int]
Scalar = int | float

# supported dtypes
float16 = half = np.float16
float32 = float = np.float32  # noqa: torch-like API
float64 = double = np.float64
ALL_DTYPES = [float16, float32, float64]
DEFAULT_DTYPE = float32

# tensor representation
PRECISION = 4
