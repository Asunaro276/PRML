import numpy as np


class SquaredError:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        self.x = None

    def _sum_squared(self, x, t):
        return (1/2) * ((x - t)**2).sum()

    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = self._sum_squared(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
