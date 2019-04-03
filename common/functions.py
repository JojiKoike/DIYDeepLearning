import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Function
    :param x: Input
    :return: Activated Value
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax Activate Function
    :param x:
    :return:
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y: np.ndarray = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
