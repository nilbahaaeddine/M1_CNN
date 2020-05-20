# Magiefication
The convolutional neural network (designated by the acronym CNN) is one of the most efficient models for performing image recognition and classification, object detection and face recognition.

## Authors
GoFocus members :
* Meriem AMERAOUI
* Dounia BELABIOD
* Jihene BOUHLEL
* Bahaa Eddine NIL

## Requirements
Python 3 is used during development and the following libraries are required to run the code provided in the notebook :
* NumPy = 1.17.2
* Matplotlib = 3.1.1
* Keras = 2.3.1
* TensorFlow = 2.2.0

## Notes
The `classes` folder contains all the classes we've implemented.
* `conv.py` : A convolution layer using 3x3 filters.
* `convolutionalneuralnetwork.py` : Main class of our network.
* `dense.py` : A standard fully-connected layer with softmax activation.
* `dropout.py` : A dropout layer to avoid the model of over learning.
* `flatten.py` : A flattening layer.
* `maxpool.py` : A max pooling layer using a pool size of 2.
* `relu.py` : Rectified linear units activation function.
---
We have 3 notebooks in this project :
* `ConvolutionalNeuralNetwok.ipynb` : Our main file where we display the data and create the model to classify images.
* `DataGenerator.ipynb` : In this file we generate more data from the same image.
* `KerasModel.ipynb` : Creation of the Keras model to classify images.
---
The `requirements.txt` file lists all the Python libraries on which the notebooks of this project depend, you can install them using :
```
pip install -r requirements.txt
```
---
The `ressouces` folder contains all the secondary files necessary for the project (Gantt diagram, report, video presentation and slides).


## Keywords
`Deep learning, convolutional neural network.`