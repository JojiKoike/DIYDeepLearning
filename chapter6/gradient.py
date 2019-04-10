from typing import Dict, Optional
import numpy as np



class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr: float = lr
        self.momentum: float = momentum
        self.v: Optional[Dict[str, np.ndarray]] = None

    def update(self, params: Dict[str, np.ndarray], grads: np.ndarray):
        if self.v is None:
            self.v = {}
            for key, val in params:
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.h: Optional[Dict[str, np.ndarray]] = None

    def update(self, params: Dict[str, np.ndarray], grads: np.ndarray):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
