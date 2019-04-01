from dataclasses import dataclass
from common.functions import softmax, cross_entropy_error
import numpy as np


@dataclass
class Relu:
    mask: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out: np.ndarray = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_out[self.mask] = 0
        dx: np.ndarray = d_out
        return dx


@dataclass
class Sigmoid:
    out: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        dx: np.ndarray = d_out * self.out * (1.0 - self.out)
        return dx


class Affine:
    def __init__(self, w: np.ndarray, b: np.ndarray):
        self.w: np.ndarray = w
        self.b: np.ndarray = b

        self.x: np.ndarray = None
        self.x_original_shape: np.ndarray = None
        self.d_w: np.ndarray = None
        self.d_b: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_original_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.w) + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_x: np.ndarray = np.dot(d_out, self.w.T)
        self.d_w = np.dot(self.x.T, d_out)
        self.d_b = np.sum(d_out, axis=0)

        d_x = d_x.reshape(*self.x_original_shape)
        return d_x


@dataclass
class SoftmaxWithLoss:
    loss: float
    output_softmax: np.ndarray
    teacher: np.ndarray

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.teacher = t
        self.output_softmax = softmax(x)
        self.loss = cross_entropy_error(self.output_softmax, self.teacher)

        return self.loss

    def backward(self, dout=1) -> np.ndarray:
        batch_size = self.teacher.shape[0]
        if self.teacher.size == self.output_softmax.shape[0]:
            dx: np.ndarray = (self.output_softmax - self.teacher) / batch_size
        else:
            dx = self.output_softmax.copy()
            dx[np.arange(batch_size), self.teacher] -= 1
            dx = dx / batch_size

        return dx
