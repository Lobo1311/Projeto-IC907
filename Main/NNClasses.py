import numpy as np
from BaseClasses import Layer, Loss, Optimizer

class Activation_ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray, isTraining:bool=True):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues.copy() #* Copy to avoid modifying the original array
        self.dinputs = np.where(self.inputs <= 0, 0, self.dinputs) #* Zero gradient where output was less than or equal to 0

class Activation_LeakyReLU(Layer):
    def __init__(self):
        super().__init__()
        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.where(inputs > 0, inputs, inputs * 0.01)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues.copy() #* Copy to avoid modifying the original array
        self.dinputs = np.where(self.inputs > 0, self.dinputs, self.dinputs * 0.01)

class Activation_Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.tanh(inputs)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues * (1 - self.output ** 2)

class Activation_Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Layer_Dense(Layer):
    def __init__(self, n_inputs:int, n_neurons:int):
        super().__init__()

        self.dweights:np.ndarray = np.array([])
        self.dbiases:np.ndarray = np.array([])

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0.01 * np.random.randn(1, n_neurons)

        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray, isTraining:bool=True):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = inputs @ self.weights + self.biases

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues @ self.weights.T
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

class Layer_Dropout(Layer):
    def __init__(self, dropout_prob: float = 0.4):
        super().__init__()

        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("Dropout probability must be between 0 and 1.")

        self.dropout_prob: float = dropout_prob
        self.dropout_mask: float = None

        self.DeactivateAttr()

    def forward(self, inputs:np.ndarray, isTraining:bool=True):
        self.inputs = inputs #* Save inputs for backpropagation
        self.dropout_mask = np.random.binomial(1, (1-self.dropout_prob), size=inputs.shape) / (1-self.dropout_prob) if isTraining else np.ones_like(inputs)
        self.output = inputs * self.dropout_mask

    # Backward pass
    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues * self.dropout_mask

class Loss_MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()
        self.DeactivateAttr()

    def forward(self, y_pred:np.ndarray, y_true:np.ndarray):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
    
        return sample_losses

    def backward(self, y_pred:np.ndarray, y_true:np.ndarray):
        nsamples = y_pred.shape[0]
        noutputs = y_pred.shape[1]

        self.dinputs = -2 * (y_true - y_pred) / noutputs #* Gradient of outputs
        self.dinputs = self.dinputs / nsamples #* normalized by the samples

class Optimizer_SGD(Optimizer):
    def __init__(self, learning_rate:float=1.0, decay_rate:float=0.0, decay_step:int=100000, momentum:float=0.0):
        super().__init__()

        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = 100000
        self.interation = 0
        self.momentum = momentum

        self.DeactivateAttr()

    def pre_update_params(self):
        if self.decay_rate:
            self.learning_rate  = self.initial_learning_rate * ( self.decay_rate **( self.interation // self.decay_step))

    def update_params(self, layer:Layer_Dense):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.ActivateAttr()
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.DeactivateAttr()

            weight_updates = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.interation += 1