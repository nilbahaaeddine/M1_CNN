import numpy as np

class Relu:
    def __init__(self):
        self.params = []

    def forward(self, input):
        self.last_input = input
        output = np.maximum(0, input)  # Element-wise
        return output

    def backprop(self, dOut):
        '''
        f′(x) = {1 if x > 0}
                {0 otherwise}
        '''
        dOut[dOut <= 0] = 0
        dOut[dOut > 0] = 1
        
        return dOut