if __name__ != '__main__':
    from .tensor import tensor
    from .autograd import AccumulateGrad  # TODO: remove

    from numpy import float16 as float16
    from numpy import float16 as half

    from numpy import float32 as float32
    from numpy import float32 as float  # noqa: torch like API

    from numpy import float64 as float64
    from numpy import float64 as double
else:
    raise ImportError('Package startup error')
