import math
import random
from .module import Module

class MSELoss:
    def __call__(self, predicted,target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p-t)**2 for p,t in zip(predicted,target)) /n
        return self.loss
    def backward(self):
        n = len(self.predicted)
        return [2*(p-t)/n for p,t in zip(self.predicted,self.target)]

class BCELoss:
    def __call__(self, predicted,target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p,t in zip(predicted,target):
            p = max(eps,min(1-eps,p))
            self.loss += -(t*math.log(p) +(1-t) * math.log(1-p))
        self.loss /= n 
        return self.loss
    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p,t in zip(self.predicted,self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads