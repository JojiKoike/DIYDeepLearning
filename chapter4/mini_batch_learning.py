import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_size: int = x_train.shape[0]
batch_size: int = 10
batch_mask: np.ndarray = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    Cross Entropy Error
    :param y: Neural Network Calculation Output
    :param t: Teacher Data
    :return: Cross Entropy Error
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size_local = y.shape[0]
    if t.shape[0] == 1:
        # one_hot expression
        return -np.sum(y[np.arange(batch_size_local), t]) / batch_size_local
    else:
        # Non one_hot expression
        return -np.sum(t * np.log(y)) / batch_size_local


