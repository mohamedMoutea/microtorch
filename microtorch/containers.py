import math 
import random
from .module import Module


class Sequential(Module):
    def __init__(self,*modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x 
    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad
    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
    def train(self):
        self.training = True 
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()