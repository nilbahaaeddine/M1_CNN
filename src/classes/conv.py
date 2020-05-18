import numpy as np

class Conv3x3: # A Convolution layer using 3x3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.input = 0
        self.output = 0

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j                

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        self.input = input
        if(input.ndim == 2): # First conv we catch the image (2D)
            self.last_input = input
            h, w = input.shape
            output = np.zeros((h - 2, w - 2, self.num_filters))
            for im_region, i, j in self.iterate_regions(input):
                output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        else: # Second conv we catch the output of maxpool (3D)
            self.last_input = input
            h, w , filters = input.shape
            output = np.zeros((h - 2, w - 2, self.num_filters))
            temp=np.transpose(input)
            for k in range(0, filters):
                for im_region, i, j in self.iterate_regions(temp[k]):
                    output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        
        self.output = output
        return output
    
    def iterate_regions_back(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        if(self.last_input.ndim == 2): # 1er conv on recup l'image 2D
            d_L_d_filters = np.zeros(self.filters.shape)

            for im_region, i, j in self.iterate_regions(self.last_input):
                for f in range(self.num_filters):
                    d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

            # Update filters
            self.filters -= learn_rate * d_L_d_filters
            # We aren't returning anything here since we use Conv3x3 as
            # the first layer in our CNN. Otherwise, we'd need to return
            # the loss gradient for this layer's inputs, just like every
            # other layer in our CNN.
            return None

        else:
            d_L_d_input = np.zeros(self.last_input.shape)

            for im_region, i, j in self.iterate_regions_back(self.last_input):
                h, w, f = im_region.shape
                amax = np.amax(im_region, axis=(0, 1))

                for i2 in range(h):
                    for j2 in range(w):
                        for f2 in range(f):
                            # If this pixel was the max value, copy the gradient to it.
                            if im_region[i2, j2, f2] == amax[f2]:
                                d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
            return d_L_d_input