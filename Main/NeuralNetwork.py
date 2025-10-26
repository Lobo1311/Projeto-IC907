import numpy as np

class Activation_ReLU:
  # Forward pass
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs

    # Calculate output values from inputs
    self.output = np.maximum(0, inputs)

  # Backward pass
  def backward(self, dvalues):
    # Gradient of ReLU is 1 for positive inputs, 0 for negative inputs
    self.dinputs = dvalues.copy() # Copy to avoid modifying the original array

    # Zero gradient where output was less than or equal to 0
    self.dinputs = np.where(self.inputs <= 0, 0, self.dinputs)


class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    # self.biases = np.zeros((1, n_neurons))
    self.biases = 0.01 * np.random.randn(1, n_neurons)

  # Forward pass
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs

    # Calculate output values from inputs, weights and biases
    self.output = inputs @ self.weights + self.biases

  # Backward pass
  def backward(self, dvalues):
    self.dinputs = dvalues @ self.weights.T
    self.dweights = self.inputs.T @ dvalues
    self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)


# Father class (for any type of loss)
class Loss:
  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):

    # Calculate sample losses
    sample_losses = self.forward(output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Return loss
    return data_loss


# MSE loss class
class Loss_MeanSquaredError(Loss):
  def forward(self, y_pred, y_true):
    # Calculate loss
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

    return sample_losses

  def backward(self, y_pred, y_true):
    nsamples = y_pred.shape[0]
    noutputs = y_pred.shape[1]

    # Recall
    # Loss = 1/nsamples * sum(L_i)
    # L_i = 1/noutputs * sum((y_true_i - y_pred_i)**2)
    self.dinputs = -2 * (y_true - y_pred) / noutputs # Gradient of outputs
    self.dinputs = self.dinputs / nsamples # normalized by the samples


# SGD optimizer
class Optimizer_SGD:
  # Initialize optimizer - set settings,
  # learning rate of 1. is default for this optimizer
  def __init__(self, learning_rate=1.0):
    self.learning_rate = learning_rate
    # Update parameters
  def update_params(self, layer):
    layer.weights -= self.learning_rate * layer.dweights
    layer.biases -= self.learning_rate * layer.dbiases