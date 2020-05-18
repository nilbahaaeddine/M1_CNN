import numpy as np
import time

from classes.conv import Conv3x3
from classes.relu import Relu
from classes.maxpool import MaxPool2
from classes.dropout import Dropout
from classes.flatten import Flatten
from classes.dense import Dense

class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.nbLayers = 0
        self.layers = []
        
    def info(self):
        for i in range(len(self.layers)):
            print(f'\tLayer #{i + 1} => {type(self.layers[i])}')
            if(type(self.layers[i]) is Dense):
                print(f'\t\tThis layer uses the {self.layers[i].activation_func} function as its activation')
        
    def addLayer(self, layer):
        self.nbLayers += 1
        self.layers.append(layer)
        
    def forward_propagation(self, X):
        outPrevious = 0
        for num_layer in range (0, self.nbLayers):
            if(type(self.layers[num_layer]) is Conv3x3):
                if num_layer == 0:
                    X = (X / 255) - 0.5
                    outPrevious = self.layers[num_layer].forward(X)
                else: 
                    outPrevious = self.layers[num_layer].forward(outPrevious)
            
            if(type(self.layers[num_layer]) is Relu):
                outPrevious = self.layers[num_layer].forward(outPrevious)
                      
            if(type(self.layers[num_layer]) is MaxPool2):
                outPrevious = self.layers[num_layer].forward(outPrevious) 
              
            if(type(self.layers[num_layer]) is Dropout):
                outPrevious = self.layers[num_layer].forward(outPrevious)
                  
            if(type(self.layers[num_layer]) is Flatten):
                outPrevious = self.layers[num_layer].forward(outPrevious) 
                
            if(type(self.layers[num_layer]) is Dense):
                outPrevious = self.layers[num_layer].forward(outPrevious)
        return outPrevious

    def cost_function(self, out, y):
        return (-np.log(out[y]))

    def backward_propagation(self, out, y, eta):
        previousGradient = 0
        for num_layer in range (self.nbLayers - 1, 0, -1):
            if(type(self.layers[num_layer]) is Dense):
                # Init the gradient at the end
                gradient = np.zeros(10)
                gradient[y] = -1 / out[y]
                previousGradient = self.layers[num_layer].backprop(gradient, eta)
            if(type(self.layers[num_layer]) is Flatten):
                previousGradient = self.layers[num_layer].backprop(previousGradient)
            if(type(self.layers[num_layer]) is Dropout):
                previousGradient = self.layers[num_layer].backprop(previousGradient)
            if(type(self.layers[num_layer]) is MaxPool2):
                previousGradient = self.layers[num_layer].backprop(previousGradient)
            if(type(self.layers[num_layer]) is Relu):
                previousGradient = self.layers[num_layer].backprop(previousGradient)
            if(type(self.layers[num_layer]) is Conv3x3):
                previousGradient = self.layers[num_layer].backprop(previousGradient, eta)

    def convert_prob_into_class(self, probs):
        probs = np.copy(probs) # To not to lose props, i.e. y_hat
        probs[probs > 0.5] = 1
        probs[probs <= 0.5] = 0
        return probs

    def accuracy(self, out, y):
        acc = 1 if np.argmax(out) == y else 0
        return acc       

    def predict(self, X):
        outPrevious = self.forward_propagation(X)
        return outPrevious

    def fit(self, X, y, *args, **kwargs):    
        epochs = kwargs.get("epochs", 20)
        verbose = kwargs.get("verbose", False)
        eta = kwargs.get("eta", 0.01)
        cost_history = []
        accuracy_history = []
        t = np.zeros(epochs)
        startTime = time.time()
        for nb_epochs in range(epochs):
            cost_all_images = []
            acc_all_images = []
            cost = 0
            acc = 0
            if(verbose is True):
                print(f'\tRunning epoch {nb_epochs + 1}...')
            for i, (im, label) in enumerate(zip(X, y)):
                # Do a forward pass.
                out = self.forward_propagation(im)
                cost += self.cost_function(out, label)
                cost_all_images.append(self.cost_function(out, label))
                acc += self.accuracy(out, label)
                acc_all_images.append(self.accuracy(out, label))
                self.backward_propagation(out, label, eta)
                if(verbose is True):
                    if i % 100 == 99:
                        endTime = time.time()
                        print(f'\t\t(Past 100 steps) Step {i + 1} : '
                              f'Average Loss : {np.float64(cost / 100):.2f} '
                              f'| Accuracy : {acc}% '
                              f'| Time : {endTime - startTime:.2f}')
                        cost = 0
                        acc = 0
                        startTime = time.time()
            current_cost = np.average(cost_all_images)
            cost_history.append(current_cost)  
            current_acc = np.average(acc_all_images)
            accuracy_history.append(current_acc)
        return cost_history, accuracy_history