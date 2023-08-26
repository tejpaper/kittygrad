from .tensor import Tensor

import abc
import numpy as np


class Function(abc.ABC):
    pass  # TODO


class AccumulateGrad:
    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    def accumulate(self, grad: np.ndarray):
        if self.tensor.shape != grad.shape:
            raise RuntimeError(f'The size of tensor {self.tensor.shape} '
                               f'must match the size of its gradient {grad.shape}')

        if self.tensor.grad is None:
            self.tensor.grad = grad
        else:
            self.tensor.grad += grad


class BackwardAccess(abc.ABC):
    def __init__(self):
        self.ctx = None
        self.next_functions = []

    @abc.abstractmethod
    def propagate(self):
        pass


class AddBackward(BackwardAccess):
    def propagate(self):
        pass  # TODO
