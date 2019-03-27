import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Function
    :param x: Input
    :return: Activated Value
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(a: np.ndarray) -> np.ndarray:
    """
    Softmax Activate Function
    :param a:
    :return:
    """
    return np.exp(a) / np.sum(np.exp(a))


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
