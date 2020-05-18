import numpy as np

class Dropout:
    def __init__(self, drop_probability):
        self.drop_probability = drop_probability
        self.mask = None

    def forward(self, input):
        mask = np.random.binomial(1, self.drop_probability, size=input.shape) / self.drop_probability
        self.mask = mask
        out = input * mask
        return out

    def backprop(self, dOut):
        return dOut - dOut * self.mask