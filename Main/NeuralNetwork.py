from BaseClasses import BasicData, Layer, Loss, Optimizer
from NNClasses import Layer_Dense, Loss_MeanSquaredError, Optimizer_SGD
from Main.DataSet import DataSet
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

        self.LossVecTrain:np.ndarray = np.zeros(self.epochs)
        self.LossVecTest:np.ndarray = None

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
    
    def train(self, trainData:DataSet, testData:DataSet=None):
        if testData: self.LossVecTest = np.zeros(self.epochs)
        
        for epoch in range(self.epochs):
            # Forward pass
            trainOutput = self.forward(trainData.x)

            # Compute loss
            trainLoss = self.Loss.calculate(trainOutput, trainData.y)
            self.LossVecTrain[epoch] = trainLoss

            # Backward pass
            self.Loss.backward(trainOutput, trainData.y)
            self.backward(self.Loss.dinputs)

            # Parameters update
            for layer in self.layers:
                if isinstance(layer, Layer_Dense):
                    self.optimizer.update_params(layer)

            if testData:
                testOutput = self.forward(testData.x)
                testLoss = self.Loss.calculate(testOutput, testData.y)
                self.LossVecTest[epoch] = testLoss

            # Print 10 times during training
            if epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}: Loss = {trainLoss:.10f}", end='')
                if testData:
                    print(f", Test Loss = {testLoss:.10f}")
                else:
                    print()

    def plot_loss(self):
        plt.plot(self.LossVecTrain, label="Train Loss")
        if self.LossVecTest is not None: plt.plot(self.LossVecTest, label="Test Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


