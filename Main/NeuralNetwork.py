from BaseClasses import BasicData, Layer, Loss, Optimizer
from NNClasses import Layer_Dense, Layer_Dropout, Loss_MeanSquaredError, Optimizer_SGD
from DataSet import DataSet
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(BasicData):
    def __init__(self, inputSize:int, lossFunc=Loss_MeanSquaredError, optimizer=Optimizer_SGD, 
                 lr:float=1.0, decay_rate:float=0.0, decay_step:int=100000, epochs:int=100, l2_regularization_weight:float=1.e-3, momentum:float=0.0):
        super().__init__()
        self.layers:list[Layer] = []

        self.epochs:int = epochs
        self.Loss:Loss = lossFunc()
        self.optimizer:Optimizer = optimizer(learning_rate = lr, decay_rate=decay_rate, decay_step=decay_step, momentum=momentum)

        self.LossVecTrain:np.ndarray = np.zeros(self.epochs)
        self.LossVecTest:np.ndarray = None

        self.LastLayerSize:int = inputSize

        self.is_train_mode: bool = True
        # self.l2_regularization: bool = l2_regularization
        self.l2_regularization_weight: float = l2_regularization_weight
    
    def build_nn(self, layers:dict):
        for i in range(len(layers)):
            num_neurons = layers[f"layer_{i}"]["neurons"]
            activation = layers[f"layer_{i}"]["activation"]
            dropout = layers[f"layer_{i}"]["dropout"]
            
            self.add_layer(num_neurons, activation, dropout)

    def add_layer(self, neurons:int, activation:Layer=None, dropout:list=[False]):
        self.layers.append(Layer_Dense(self.LastLayerSize, neurons))
        self.LastLayerSize = neurons
        
        if activation: self.layers.append(activation)

        if dropout[0]: self.layers.append(Layer_Dropout(self.is_train_mode, dropout[1] if len(dropout) > 1 else None))

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
        # set train mode
        self.is_train_mode = True

        if testData: self.LossVecTest = np.zeros(self.epochs)
        
        for epoch in range(self.epochs):
            # Forward pass
            
            trainOutput = self.forward(trainData.x)

            # Backward pass
            self.Loss.backward(trainOutput, trainData.y)
            self.backward(self.Loss.dinputs)

            # include l2 regularization
            if self.l2_regularization_weight:
                for layer in self.layers:
                    if isinstance(layer, Layer_Dense):
                        layer.dweights += (2 * self.l2_regularization_weight * layer.weights)

            # Compute loss
            trainLoss = self.Loss.calculate(trainOutput, trainData.y)
            self.LossVecTrain[epoch] = trainLoss

            # Parameters update
            self.optimizer.pre_update_params()

            for layer in self.layers:
                if isinstance(layer, Layer_Dense):
                    self.optimizer.update_params(layer)

            self.optimizer.post_update_params()

            if testData:
                testOutput = self.forward(testData.x)
                testLoss = self.Loss.calculate(testOutput, testData.y)
                self.LossVecTest[epoch] = testLoss

            # Print 10 times during training


            # print(epoch)

            if trainLoss != trainLoss:  # Check for NaN
                raise ValueError("Loss is NaN. Try adjusting the learning rate or check for issues in the data/model.")

            if epoch % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: lr = {self.optimizer.learning_rate}, Loss = {trainLoss:.10f}", end='')
                if testData:
                    print(f", Test Loss = {testLoss:.10f}")
                else:
                    print()
            
        # deactivate train mode
        self.is_train_mode = False

    def predict(self, inputs:np.ndarray):
        # deactivate the train mode
        self.is_train_mode = False
        # foward the inputs
        y_pred: np.ndarray = self.forward(inputs=inputs)

        return y_pred

    def plot_loss(self):
        plt.plot(self.LossVecTrain, label="Train Loss")
        if self.LossVecTest is not None: plt.plot(self.LossVecTest, label="Test Loss")
        
        
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        x_epochs = np.arange(self.epochs)   
        plt.plot(x_epochs[-int(len(x_epochs)/10):], self.LossVecTrain[-int(len(self.LossVecTrain)/10):], label="Train Loss")
        if self.LossVecTest is not None: plt.plot(x_epochs[-int(len(x_epochs)/10):], self.LossVecTest[-int(len(self.LossVecTest)/10):], label="Test Loss")
        
        
        plt.title("Loss over Epochs (Last 10%)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


        if self.LossVecTest is not None:
            ratio_list = self.LossVecTest / self.LossVecTrain
            plt.plot(ratio_list, label="Test/Train Loss Ratio")
            plt.title("Test/Train Loss Ratio over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Ratio")
            plt.legend()
            plt.show()


        