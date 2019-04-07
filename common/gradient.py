from typing import Callable, Tuple
import numpy as np


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    h: float = 1e-4
    grad: np.ndarray = np.zeros_like(x)

    itr: np.nditer = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        idx: Tuple[int, int] = itr.multi_index
        tmp_value: float = x[idx]
        x[idx] = tmp_value + h
        fxh1: np.ndarray = f(x)

        x[idx] = tmp_value - h
        fxh2: np.ndarray = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_value
        itr.iternext()

    return grad
