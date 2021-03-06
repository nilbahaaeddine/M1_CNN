class Flatten: # A Flattening layer
    def forward_propagation(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return input
    
    def backward_propagation(self, d_L_d_out):
        return d_L_d_out.reshape(self.last_input_shape)