import sys
import os
sys.path.append(os.pardir)
from typing import Dict
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
        self.lastLayer = SoftmaxWithLoss()




