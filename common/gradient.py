from typing import Callable
import numpy as np


def numerical_gradient(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    h: float = 1e-4
    grad: np.ndarray = np.zeros_like(x)

    for idx in range(x.size):
        tmp_value = x[idx]
        x[idx] = tmp_value + h
        fxh1: np.ndarray = f(x)

        x[idx] = tmp_value - h
        fxh2: np.ndarray = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_value

    return grad
