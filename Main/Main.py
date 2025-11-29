from NeuralNetwork import NeuralNetwork
from NNClasses import *
from DataSet import DataSet
from NeuralNetwork_torch import ANN
from LossPINN import LossPINN

import numpy as np
import matplotlib.pyplot as plt

from DarcyTransientFlow import DarcyTransientFlow

def main_nn_by_hand():
    # Real a and b for the line
    seed = 0
    np.random.seed(seed)

    xpts = np.linspace(-2, 3, 100)
    set_xnew = np.linspace(-2, 3, 1000)
    npts = len(xpts)

    # Generating data
    yreal = (
            np.sin(12*np.pi*xpts)
            + 0.3*np.cos(4*np.pi*xpts)      
            + np.sin(5*np.pi*xpts - 0.1)
        ) +  0.1*np.random.randn(npts)
    
    yreal = (
            np.sin(xpts*np.pi)
            # +  0.3*np.random.randn(npts)
            # + 0.3*np.cos(4*np.pi*xpts)      
           
        )
    
    yreal = np.where(np.sin(15 * xpts) + 0.3 * xpts**2 > 0, 1, 0)
    # yreal = (np.sin(10*xpts)) ## Second function


    Data = DataSet(xpts.reshape(npts, 1), yreal.reshape(npts, 1))
    train_set, test_set = Data.split(0.7) 

    # Learning rate and number of epochs
    lr = 0.1
    epochs = 100000
    # epochs = 40000
    decay_rate = 0  # 0.0 for no decay
    decay_step = 100000

    momentum = 0.00  # 0.0 for no momentum
    l2_regularization_weight = 0.0  # 0.0 for no L2 regularization
    
    # Creating the neural network
    nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, decay_step=decay_step, optimizer=Optimizer_SGD, l2_regularization_weight=l2_regularization_weight, momentum=momentum)
    # nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, optimizer=Optimizer_SGD_Decay, l2_regularization=False, l2_regularization_weight=1.e-4)
    # nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, optimizer=Optimizer_SGD, l2_regularization=False, l2_regularization_weight=1.e-3)

    nn_layers = {
                "layer_0": 
                    {
                        "neurons": 500, 
                        "activation": Activation_ReLU(),
                        "dropout": [True, 0.5],
                    },
                "layer_1": 
                    {
                        "neurons": 400, 
                        "activation": Activation_ReLU(),
                        "dropout":  [True, 0.5],
                    },
                "layer_2": 
                    {
                        "neurons": 200, 
                        "activation": Activation_ReLU(),
                        "dropout":  [False, 0.5],
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
    nn.plot_loss()

    # plt.scatter(xpts, yreal, label='True data')

    plt.plot(xpts, yreal, '-', color='orange', label='True function')
    plt.scatter(train_set.x, train_set.y, s = 20, label='Train set')
    plt.scatter(test_set.x, test_set.y, s = 20, label='Test set')

    # plt.scatter(test_set.x, test_set.y, label='Test set')

    y_pred = nn.predict(set_xnew.reshape(-1, 1))
    plt.plot(set_xnew, y_pred.flatten(),  color = 'green', label='NN prediction')

 
    plt.title(f'Fit with neural net')
    plt.legend()
    plt.show()

    a = 1

    
def main_nn_torch():
    
    # Real a and b for the line
    seed = 42
    np.random.seed(seed)

    darcy_func: callable = ... # buid as a lambda func?
    
    xpts = np.linspace(0, 1, 70)
    ypts = darcy_func(xpts)

    # set the data set
    data_set = DataSet(xpts, ypts)
    data_set.add_noise(randomization_factor=1.)
    
    train_set, test_set = data_set.split(0.5) # maior numero de pontos de treino

    # initialize pinn loss funcs
    my_pinn_loss: LossPINN = LossPINN() # the loss funcs must be change inside this class... it can be improved later

    # set hyperparameters
    lr = 0.01
    epochs = 300000
    input_dim = 1
    output_dim = 1
    hidden_layers = [200]

    model = ANN(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, loss2=my_pinn_loss.bc_loss_fn, loss2_weight=1e0)

    # model.train_nn(x, y)

    # model.plot_loss()
    # model.plot_prediction(x_limits=, x_training_data=, y_training_data=, analitycal_func=darcy_func)
    ...

def EquationTest():
    L = 1.0
    PL = 1e5      #* Left boundary pressure (Pa)
    PR = 0.0      #* Right boundary pressure (Pa)
    k = 1e-12    #* Permeability (m^2)
    mu = 1e-3     #* Dynamic viscosity (Pa.s)
    phi = 0.2     #* Porosity (-)
    ct = 1e-9     #* Total compressibility (1/Pa)

    problem = DarcyTransientFlow(L, PL, PR, k, mu, phi, ct)

    problem.Plot()


if __name__ == "__main__":
    # main_nn_by_hand() # use to nn by hand validation
    # main_nn_torch() # use to PINN validation
    EquationTest()