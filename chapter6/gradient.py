from typing import Dict
import numpy as np


class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
