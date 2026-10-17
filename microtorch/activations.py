from .module import Module
import math
import random

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None
    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0,v) for v in x]
    def backward(self, grad):
        return [g*m for g,m in zip(grad,self.mask)]

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None
    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500,min(500,v))
            self.output.append(1.0/(1.0 + math.exp(-v)))
        return self.output
    def backward(self, grad):
        return [g * o * (1-o) for g,o in zip(grad,self.output)]

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output
    def backward(self, grad):
        return [g * (1-o*o) for g,o in zip(grad,self.output)]
    