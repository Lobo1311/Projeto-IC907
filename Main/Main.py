from NeuralNetwork import NeuralNetwork
from NNClasses import Layer_Dense, Activation_ReLU
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Real a and b for the line
    xpts = np.linspace(0, 1, 100)
    npts = len(xpts)

    # Generating data
    yreal = np.sin(2*np.pi*xpts)

    # Learning rate and number of epochs
    lr = 0.1
    epochs = 10001

    # Creating the neural network
    nn = NeuralNetwork(1, lr=lr, epochs=epochs)
    nn.add_layer(50, Activation_ReLU())
    nn.add_layer(25, Activation_ReLU())
    nn.add_layer(1)

    # Training the neural network
    nn.train(xpts.reshape(npts, 1), yreal.reshape(npts, 1))

    # Plotting results
    nn.plot_loss()

    plt.plot(xpts, yreal, 'o', label='True data')
    plt.plot(xpts, nn.layers[-1].output, '-', label='NN prediction')
    plt.title('Sine fit with neural net')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()