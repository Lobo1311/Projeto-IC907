import numpy as np
from BaseClasses import Layer, Loss, Optimizer

class Activation_ReLU(Layer):
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues.copy() #* Copy to avoid modifying the original array
        self.dinputs = np.where(self.inputs <= 0, 0, self.dinputs) #* Zero gradient where output was less than or equal to 0

class Activation_LeakyReLU(Layer):
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.where(inputs > 0, inputs, inputs * 0.01)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues.copy() #* Copy to avoid modifying the original array
        self.dinputs = np.where(self.inputs > 0, self.dinputs, self.dinputs * 0.01)

class Activation_Tanh(Layer):
    def forward(self, inputs:np.ndarray):
        self.inputs = inputs #* Save inputs for backpropagation
        self.output = np.tanh(inputs)

    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues * (1 - self.output ** 2)

class Activation_Sigmoid(Layer):
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

class Layer_Dropout(Layer):
    def __init__(self, is_training_mode: bool = False, dropout_prob: float = None):
        super().__init__()

        self.is_training_mode: bool = is_training_mode
        
        self.dropout_prob: float = dropout_prob if dropout_prob is not None else 0.4

        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("Dropout probability must be between 0 and 1.")
        
        self.dropout_mask: float = None

    def forward(self, inputs:np.ndarray):
        if self.is_training_mode:

            self.dropout_mask = np.random.binomial(1, (1-self.dropout_prob), size=inputs.shape) / (1-self.dropout_prob)
            self.inputs = inputs #* Save inputs for backpropagation
            self.output = inputs * self.dropout_mask

        else:
            self.dropout_mask = np.ones_like(inputs)

            self.inputs = inputs #* Save inputs for backpropagation
            self.output = inputs

    # Backward pass
    def backward(self, dvalues:np.ndarray):
        self.dinputs = dvalues * self.dropout_mask

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

    def pre_update_params(self):
        pass

    def update_params(self, layer:Layer_Dense):

        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

    def post_update_params(self):
        pass

class Optimizer_SGD_Decay(Optimizer):
    def __init__(self, learning_rate:float=1.0, decay_rate:float=0.8):
        super().__init__()

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.step = 0

    def pre_update_params(self):
        
        self.current_learning_rate  = self.learning_rate * (1 / ( 1 + self.decay_rate * self.step))

    def update_params(self, layer:Layer_Dense):

        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases -= self.current_learning_rate * layer.dbiases
    
    def post_update_params(self):
        self.step += 1