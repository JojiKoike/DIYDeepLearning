import sys
import os
sys.path.append(os.pardir)
from common.functions import sigmoid, softmax, cross_entropy_error
from common.gradient import numerical_gradient
from typing import Dict, Callable, NoReturn
import numpy as np


class TwoLayerNet:
    """
    Two Layer Neural Network
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, weight_init_std: float = 0.01) -> NoReturn:
        self.params: Dict[str, np.ndarray] = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        W1: np.ndarray
        W2: np.ndarray
        b1: np.ndarray
        b2: np.ndarray
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1: np.ndarray = np.dot(x, W1) + b1
        z1: np.ndarray = sigmoid(a1)
        a2: np.ndarray = np.dot(z1, W2) + b2
        y: np.ndarray = softmax(a2)

        return y

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y: np.ndarray = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y: np.ndarray = self.predict(x)
        y = np.argmax(y, axis=1)
        t_max: np.ndarray = np.argmax(t, axis=1)

        accuracy: float = np.sum(y == t_max) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        loss_w: Callable[[np.ndarray], np.ndarray] = lambda w: self.loss(x, t)

        grad: Dict[str, np.ndarray] = {}
        grad['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grad

    def gradient(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        pass
