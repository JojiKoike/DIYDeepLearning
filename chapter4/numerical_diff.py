from typing import Callable
import numpy as np


def function_2(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def numerical_gradient(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    h: float = 1e-4
    grad: np.ndarray = np.zeros_like(x)

    for idx in range(x.size):
        tmp: float = x[idx]
        x[idx] = tmp + h
        fxh1: np.ndarray = f(x)

        x[idx] = tmp - h
        fxh2: np.ndarray = f(x)

        grad[idx] = (fxh1 - fxh2) / (2.0 * h)
        x[idx] = tmp

    return grad


if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))

