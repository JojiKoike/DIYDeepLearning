import sys
import os
sys.path.append(os.pardir)
from typing import Callable
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        z: np.ndarray = self.predict(x)
        y: np.ndarray = softmax(z)
        loss: float = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net: SimpleNet = SimpleNet()
    print(net.W)

    x: np.ndarray = np.array([0.6, 0.9])
    p: np.ndarray = net.predict(x)
    print(p)
    max_idx: int = np.argmax(p)
    t = np.array([0, 0, 0])
    t[max_idx] = 1
    print(net.loss(x, t))

    f: Callable[[np.ndarray], np.ndarray] = lambda w: net.loss(x, t)
    dw: np.ndarray = numerical_gradient(f, x)
    print(dw)
