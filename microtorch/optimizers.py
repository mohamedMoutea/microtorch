import math
import random


class SGD:
    def __init__(self,paramerters,lr=0.01):
        self.params = paramerters
        self.lr = lr

    def step(self):
        for container,i,j,grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container,i,j,grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] =0.0


class Adam:
    def __init__(self,parameters,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1=beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx,(container,i,j,grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] +(1-self.beta1) *g  
            self.v[idx] = self.beta2 * self.v[idx] +(1-self.beta2) *g * g

            m_hat = self.m[idx]/(1-self.beta1 ** self.t)
            v_hat = self.v[idx]/(1-self.beta2 ** self.t)


            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update
    def zero_grad(self):
         for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
