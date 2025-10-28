import numpy as np
from Main.BaseClasses import Layer, Loss, Optimizer

class Activation_ReLU(Layer):
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues.copy() #* Copy to avoid modifying the original array
        self.dinputs = np.where(self.inputs <= 0, 0, self.dinputs) #* Zero gradient where output was less than or equal to 0

class Layer_Dense(Layer):
    def __init__(self, n_inputs:int, n_neurons:int):
        super().__init__()

        self.dweights:np.ndarray = None
        self.dbiases:np.ndarray = None

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #? self.biases = np.zeros((1, n_neurons))
        self.biases = 0.01 * np.random.randn(1, n_neurons)

    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = inputs @ self.weights + self.biases

    # Backward pass
    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues @ self.weights.T
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

class Loss_MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred:np.ndarray, y_true:np.ndarray):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, y_pred:np.ndarray, y_true:np.ndarray):
        nsamples = y_pred.shape[0]
        noutputs = y_pred.shape[1]

        self.dinputs = -2 * (y_true - y_pred) / noutputs #* Gradient of outputs
        self.dinputs = self.dinputs / nsamples #* normalized by the samples

class Optimizer_SGD(Optimizer):
    def __init__(self, learning_rate:float=1.0):
        super().__init__()

        self.learning_rate = learning_rate

    def update_params(self, layer:Layer_Dense):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases