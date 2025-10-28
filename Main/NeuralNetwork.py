from BaseClasses import BasicData
import numpy as np

class NeuralNetwork(BasicData):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs:np.ndarray):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, dvalues:np.ndarray):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
            
        return dvalues