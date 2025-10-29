from NeuralNetwork import NeuralNetwork
from NNClasses import Layer_Dense, Activation_ReLU
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Real a and b for the line
    xpts = np.linspace(0,1,100)
    npts = len(xpts)

    # Generating data
    yreal = np.sin(2*np.pi*xpts)

    # Learning rate and number of epochs
    lr = 0.1
    epochs = 10001

    # Creating the neural network
    nn = NeuralNetwork(lr=lr, epochs=epochs)
    nn.add_layer(Layer_Dense(1, 50))
    nn.add_layer(Activation_ReLU())
    nn.add_layer(Layer_Dense(50, 25))
    nn.add_layer(Activation_ReLU())
    nn.add_layer(Layer_Dense(25, 1))
    #todo make a better constructor

    # Training the neural network
    nn.train(xpts.reshape(npts, 1), yreal.reshape(npts, 1))

    nn.plot_loss()

    plt.plot(xpts, yreal, 'o', label='True data')
    plt.plot(xpts, nn.layers[-1].output, '-', label='NN prediction')
    plt.title('Sine fit with neural net')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()