from dataclasses import dataclass
from typing import Tuple


@dataclass
class MulLayer:

    x: float = 0.0
    y: float = 0.0

    def forward(self, x: float, y: float) -> float:
        self.x = x
        self.y = y
        return x * y

    def backward(self, d_out: float) -> Tuple[float, float]:
        dx: float = d_out * self.y
        dy: float = d_out * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x: float, y: float) -> float:
        return x + y

    def backward(self, d_out: float) -> Tuple[float, float]:
        dx = d_out * 1.0
        dy = d_out * 1.0
        return dx, dy
