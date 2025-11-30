from NeuralNetwork import NeuralNetwork
from NNClasses import *
from DataSet import DataSet
from NeuralNetwork_torch import PINN

import numpy as np
import matplotlib.pyplot as plt

from DarcyTransientFlow import DarcyTransientFlow
from NeuralNetwork_torch import LossPINN
import torch.nn as nn
import torch

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
    epochs = 1000
    # epochs = 40000
    decay_rate = 0  # 0.0 for no decay
    decay_step = 100000

    momentum = 0.00  # 0.0 for no momentum
    l2_regularization_weight = 0.0  # 0.0 for no L2 regularization
    
    # Creating the neural network
    nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, decay_step=decay_step, optimizer=Optimizer_SGD, l2_regularization_weight=l2_regularization_weight, momentum=momentum)
    #nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, optimizer=Optimizer_SGD_Decay, l2_regularization=False, l2_regularization_weight=1.e-4)
    #nn = NeuralNetwork(1, lr=lr, epochs=epochs, decay_rate=decay_rate, optimizer=Optimizer_SGD, l2_regularization=False, l2_regularization_weight=1.e-3)

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

def grad(outputs, inputs):
  return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    
def main_nn_torch():
    #* problem definition
    L = 1.0       #* Length of the domain (m)
    PL = 1e5      #* Left boundary pressure (Pa)
    PR = 0.0      #* Right boundary pressure (Pa)
    k = 1e-12    #* Permeability (m^2)
    mu = 1e-3     #* Dynamic viscosity (Pa.s)
    phi = 0.2     #* Porosity (-)
    ct = 1e-9     #* Total compressibility (1/Pa)
    t_start = 0.001  #* Initial time (s)
    t_final = 1.0  #* Final time (s)

    numpoints = 100

    problem = DarcyTransientFlow(L, PL, PR, k, mu, phi, ct, startTime=t_start, endTime=t_final)

    # xpts = np.array([[x, t] for t in np.linspace(t_start, t_final, numpoints) for x in np.linspace(0, L, numpoints)])
    # ypts = np.array([problem.AnalyticalSolution(x, t) for x, t in xpts])

    # # set the data set
    # data_set = DataSet(xpts, ypts)
    # data_set.add_noise(1.0)
    
    # train_set, test_set = data_set.split(0.5) # maior numero de pontos de treino

    #* Boundary condition P(0, t) = PL
    def loss_bc_left(model:PINN, numpoints:int=100):
        x0 = np.array([[0, t] for t in np.linspace(model.Problem.startTime, model.Problem.endTime, numpoints)])
        x0_torch = model.np_to_th(x0).requires_grad_(True)
        P0_pred = model(x0_torch)

        loss_bc_left = (P0_pred - model.Problem.PLeft)**2
        loss_bc_left = torch.mean(loss_bc_left)

        return loss_bc_left

    #* Boundary condition P(L, t) = PR
    def loss_bc_right(model:PINN, numpoints:int=100):
        xL = np.array([[model.Problem.L, t] for t in np.linspace(model.Problem.startTime, model.Problem.endTime, numpoints)])
        xL_torch = model.np_to_th(xL).requires_grad_(True)
        PL_pred = model(xL_torch)

        loss_bc_right = (PL_pred - model.Problem.PRight)**2
        loss_bc_right = torch.mean(loss_bc_right)

        return loss_bc_right

    #* Physics loss (PDE residual)
    def physics_loss(model:PINN):
        X_collocation = np.array([[x, t] for t in np.linspace(t_start, t_final, numpoints) for x in np.linspace(0, L, numpoints)])
        X_collocation_torch = model.np_to_th(X_collocation).requires_grad_(True)
        P_pred = model(X_collocation_torch)
        dPdt = grad(P_pred, X_collocation_torch)[:, 1:2]
        dPdx = grad(P_pred, X_collocation_torch)[:, 0:1]
        dPdx2 = grad(dPdx, X_collocation_torch)[:, 0:1]

        pde = dPdt - (model.Problem.k / (model.Problem.mu * model.Problem.phi * model.Problem.ct)) * dPdx2

        return torch.mean(pde**2)


    # set hyperparameters
    lr = 0.01
    epochs = 1000
    input_dim = 2
    output_dim = 1
    hidden_layers = [50, 50]

    LossPINNVec = []
    LossPINNVec.append(LossPINN(loss_bc_left, 0.1))
    LossPINNVec.append(LossPINN(loss_bc_right, 0.1))
    LossPINNVec.append(LossPINN(physics_loss, 0.1))

    model = PINN(input_dim, hidden_layers, output_dim, nn.ReLU, epochs, None, lr, LossPINNVec, problem)

    model.train_nn()
    model.plot_loss()
    model.plot_prediction()
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
    #main_nn_by_hand() # use to nn by hand validation
    main_nn_torch() # use to PINN validation
    #EquationTest()