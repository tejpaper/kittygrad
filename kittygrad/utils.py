import numpy as np

ALL_DTYPES = [np.float16, np.float32, np.float64]
DEFAULT_DTYPE = np.float32

Scalar = int | float  # because of numbers.Number includes complex numbers


def flatten(x: list) -> list:
    return sum(map(flatten, x), []) if isinstance(x, list) else [x]
