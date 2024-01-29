from __future__ import annotations

import math

import numpy as np

import kittygrad as kitty
from kittygrad.utils.constants import Size, Scalar, DEFAULT_DTYPE


def calculate_gain(nonlinearity: str) -> Scalar:
    match nonlinearity:
        case 'linear' | 'identity' | 'sigmoid':
            return 1
        case 'tanh':
            return 5 / 3
        case 'relu':
            return math.sqrt(2)
        case _:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}.")


def _calculate_fan_in_and_fan_out(shape: Size) -> tuple[int, int]:
    match len(shape):
        case 2:  # linear
            return shape
        case _:
            raise ValueError(f"There is no rule for determining fan in and fan out "
                             f"for a tensor with shape {shape}.")


def kaiming_uniform(shape: Size,
                    dtype: type | np.dtype | None = None,
                    requires_grad: bool = True,  # trainable parameter expected
                    nonlinearity: str = 'relu',
                    ) -> kitty.Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(shape)
    gain = calculate_gain(nonlinearity)
    bound = gain * math.sqrt(3 / fan_in)

    return kitty.tensor(
        data=np.random.uniform(-bound, bound, shape),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)


def kaiming_normal(shape: Size,
                   dtype: type | np.dtype | None = None,
                   requires_grad: bool = True,  # trainable parameter expected
                   nonlinearity: str = 'relu',
                   ) -> kitty.Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(shape)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan_in)

    return kitty.tensor(
        data=np.random.normal(scale=std, size=shape),
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
        requires_grad=requires_grad)
