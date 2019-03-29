import numpy as np
from dataclasses import dataclass


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
