import numpy as np


def smooth_curve(x: np.ndarray) -> np.ndarray:
    """
    Smooth Loss Function Graph
    :param x: Original Value
    :return: Smoothed Value
    """
    window_len: int = 11
    s: np.ndarray = np.r_[x[window_len - 1: 0: -1], x, x[-1:-window_len: -1]]
    w: np.ndarray = np.kaiser(window_len, 2)
    y: np.ndarray = np.convolve(w/w.sum(), s, mode='valid')
    return y[5: len(y)-5]
