from dataclasses import dataclass
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
