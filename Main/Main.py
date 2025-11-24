from NeuralNetwork import NeuralNetwork
from NNClasses import *
from DataSet import DataSet
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Real a and b for the line
    seed = 42
    np.random.seed(seed)

    xpts = np.linspace(0, 1, 200)
    npts = len(xpts)

    # Generating data
    yreal = (
            np.sin(2*np.pi*xpts)
            + 0.3*np.sin(12*np.pi*xpts)      
    
        )

    Data = DataSet(xpts.reshape(npts, 1), yreal.reshape(npts, 1))
    train_set, test_set = Data.split(0.3)

    # Learning rate and number of epochs
    lr = 0.15
    epochs = 10000

    # Creating the neural network
    nn = NeuralNetwork(1, lr=lr, epochs=epochs)

    nn_layers = {
                "layer_0": 
                    {
                        "neurons": 512, 
                        "activation": Activation_LeakyReLU(),
                        "dropout": [False]
                    },
                "layer_1": 
                    {
                        "neurons": 512, 
                        "activation": Activation_LeakyReLU(),
                        "dropout": [True, 0.4],
                    },
                "layer_2": 
                    {
                        "neurons": 512, 
                        "activation": Activation_Tanh(),
                        "dropout": [True, 0.4],
                    },
                "layer_3": 
                    {
                        "neurons": 1, 
                        "activation": None,
                        "dropout": [False]
                    }
                }
    
    nn.build_nn(nn_layers)
    
    # Training the neural network
    nn.train(train_set, test_set)

    # Plotting results
    # nn.plot_loss()

    # plt.plot(xpts, yreal, 'o', label='True data')
    # plt.plot(xpts, nn.layers[-1].output, '-', label='NN prediction')
    # plt.title('Sine fit with neural net')
    # plt.legend()
    # plt.show()

    plt.plot(xpts, yreal, '-', label='True data')
    plt.scatter(train_set.x, nn.forward(train_set.x), label='NN prediction')
    plt.scatter(test_set.x, nn.forward(test_set.x), label='NN test prediction')

    plt.title('Sine fit with neural net')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()