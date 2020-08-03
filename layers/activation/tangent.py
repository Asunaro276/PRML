import numpy as np


class Tangent:
    def __init__(self):
        self.h = None

    def forward(self, x):
        self.h = np.tanh(x)
        return self.h

    def backward(self, dout):
        dx = (1 - self.h**2)*dout

        return dx
