class Module:
    def __init__(self):
        self.training = True

    def forward(self,x):
        raise NotImplemented
    def backward(self,grad):
        raise NotImplemented
    def parameters(self):
        return []
    def train(self):
        self.training = True
    def eval(self):
        self.training = False