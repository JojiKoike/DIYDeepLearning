import sys
import os
sys.path.append(os.pardir)
from typing import Dict, Callable
import numpy as np
from common.layers import Affine, SoftmaxWithLoss, Sigmoid, Relu
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        self.params: Dict[str, np.ndarray] = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                                              'b1': np.zeros(hidden_size),
                                              'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                                              'b2': np.zeros(hidden_size)}

        self.layers: OrderedDict = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer: SoftmaxWithLoss = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        y: np.ndarray = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y: np.ndarray = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        loss_w: Callable[[np.ndarray], np.ndarray] = lambda w: self.loss(x, t)

        grads: Dict[str, np.ndarray] = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        # forward
        self.loss(x, t)

        # backward
        d_out = 1
        d_out = self.lastLayer.backward(d_out)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)

        grads = {}
        grads['W1'] = self.layers['Affine1'].d_w
        grads['b1'] = self.layers['Affine1'].d_b
        grads['W2'] = self.layers['Affine2'].d_w
        grads['b2'] = self.layers['Affine2'].d_b

        return grads
