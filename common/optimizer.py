import numpy as np
from typing import Dict, Optional


class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, lr: float = 0.01):
        self.lr: float = lr

    def update(self, param: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in param.keys():
            param[key] -= self.lr * grads[key]


class Momentum:
    """
    Momentum SGD
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v: Optional[Dict[str, np.ndarray]] = None

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]