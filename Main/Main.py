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
            + 0.3*np.sin(8*np.pi*xpts)      
        )
    
    # yreal = (np.sin(30*xpts) + np.cos(10*xpts) + 2 + xpts**2 ) ## Second function


    Data = DataSet(xpts.reshape(npts, 1), yreal.reshape(npts, 1))
    train_set, test_set = Data.split(0.3)

    # Learning rate and number of epochs
    lr = 0.2
    epochs = 80000
    decay_rate = 1e-3

    # Creating the neural network
    nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, optimizer=Optimizer_SGD_Decay)

    nn_layers = {
                "layer_0": 
                    {
                        "neurons": 300, 
                        "activation": Activation_LeakyReLU(),
                        "dropout": [True, 0.2],
                    },
                "layer_1": 
                    {
                        "neurons": 100, 
                        "activation": Activation_LeakyReLU(),
                        "dropout":  [False],
                    },
                # "layer_2": 
                #     {
                #         "neurons": 512, 
                #         "activation": Activation_Tanh(),
                #         "dropout": [True, 0.4],
                #     },
                "layer_2": 
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


    set_xnew = np.linspace(0, 1, 200)
    y_pred = nn.forward(set_xnew.reshape(-1, 1))
    plt.plot(set_xnew, y_pred.flatten(),  'r-',label='NN prediction')

    # plt.scatter(train_set.x, train_set.y, label='NN prediction')
   
    # plt.plot(test_set.x, (nn.forward(test_set.x.reshape(-1, 1))).flatten(), 'r-', label='NN test prediction')

    plt.title('Sine fit with neural net')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()