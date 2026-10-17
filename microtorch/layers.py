import math
import random
from .module import Module

class Linear(Module):
    def __init__(self,fan_in,fan_out):
        super().__init__()
        std = math.sqrt(2.0/fan_in)
        self.weights = [[random.gauss(0,std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self,x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self,grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights,i,j,self.weight_grads))
            params.append((self.biases,i,None,self.bias_grads))
        return params

class Dropout(Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0/(1-self.p) for _ in x]
        return [v * m for v,m in zip(x,self.mask)]
    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g*m for g,m in zip(grad,self.mask)]

class BatchNorm(Module):
    def __init__(self,size,momentum=0.1,eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self,batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m/batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch :
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v/batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v+self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch :
                normed = [(sample[j]- mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1-self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1-self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 /math.sqrt(v+self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)
        return output_batch
    def forward(self,x):
        result = self.forward_batch([x])
        return result[0]
    def backward(self,grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]
    def parameters(self):
        params=[]
        for j in range(self.size):
            params.append((self.gamma,j,None,self.gamma_grads))
            params.append((self.beta,j,None,self.beta_grads))
        return params