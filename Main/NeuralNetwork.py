from BaseClasses import BasicData, Layer, Loss, Optimizer
from NNClasses import Layer_Dense, Loss_MeanSquaredError, Optimizer_SGD
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(BasicData):
    def __init__(self, inputSize:int, lossFunc=Loss_MeanSquaredError, optimizer=Optimizer_SGD, lr:float=1.0, epochs:int=100):
        super().__init__()
        self.layers:list[Layer] = []

        self.lr:float = lr
        self.epochs:int = epochs
        self.Loss:Loss = lossFunc()
        self.optimizer:Optimizer = optimizer(lr)

        self.LossVec:np.ndarray = np.zeros(self.epochs)
        #todo: separate train and validation loss vectors

        self.LastLayerSize:int = inputSize

    def add_layer(self, neurons:int, activation:Layer=None):
        self.layers.append(Layer_Dense(self.LastLayerSize, neurons))
        self.LastLayerSize = neurons
        
        if activation: self.layers.append(activation)

    def forward(self, inputs:np.ndarray):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output

        return self.layers[-1].output

    def backward(self, dvalues:np.ndarray):
        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs
            
        return
    
    def train(self, input:np.ndarray, y_true:np.ndarray):
        for epoch in range(self.epochs):
            # Forward pass
            output = self.forward(input)

            # Compute loss
            loss = self.Loss.calculate(output, y_true)
            self.LossVec[epoch] = loss

            # Print 10 times during training
            if epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}: Loss = {loss:.10f}")

            # Backward pass
            self.Loss.backward(output, y_true)
            self.backward(self.Loss.dinputs)

            # Parameters update
            for layer in self.layers:
                if isinstance(layer, Layer_Dense):
                    self.optimizer.update_params(layer)

    def plot_loss(self): #todo: separate train and validation loss vectors
        plt.plot(self.LossVec)
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


