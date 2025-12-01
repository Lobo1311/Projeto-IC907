from NeuralNetwork import NeuralNetwork
from NNClasses import *
from DataSet import DataSet
from NeuralNetwork_torch import PINN

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    
def main_nn_torch():
    # set seed to zero
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    #* problem definition
    L = 1.0       #* Length of the domain (m)
    PL = 1.0      #* Left boundary pressure (Pa)
    PR = 0.0      #* Right boundary pressure (Pa)
    k = 1e-12    #* Permeability (m^2)
    mu = 1e-3     #* Dynamic viscosity (Pa.s)
    phi = 0.2     #* Porosity (-)
    ct = 1e-9     #* Total compressibility (1/Pa)
    t_start = 0.0001  #* Initial time (s)
    t_final = 0.1  #* Final time (s)

    numpoints = 1000

    problem: DarcyTransientFlow = DarcyTransientFlow(L, PL, PR, k, mu, phi, ct, startTime=t_start, endTime=t_final)

    # training points
    t_xpts = np.linspace(0, L, numpoints)
    t_tpts = np.linspace(t_start, t_final, numpoints)
    t_xtpts = []
    t_ypts = []
    for i in range(numpoints):
        analytical = problem.AnalyticalSolution(t_xpts[i], t_tpts[i])
        t_ypts.append(analytical)
        t_xtpts.append([t_xpts[i], t_tpts[i]])
    t_ypts = np.array(t_ypts)
    t_xtpts = np.array(t_xtpts)

    Data = DataSet(t_xtpts.reshape(numpoints, 2), t_ypts.reshape(numpoints, 1))
    Data.add_noise(0.1) 

    t_xtpts = Data.x
    t_ypts = Data.y

    #* Gradient function
    def grad(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    
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
        X_collocation = np.random.uniform(low=[0.0, model.Problem.startTime], high=[model.Problem.L, model.Problem.endTime], size=(numpoints, 2))
        X_collocation_torch = model.np_to_th(X_collocation).requires_grad_(True)

        P_pred = model(X_collocation_torch)

        # grads are the first derivatives       
        grads = grad(P_pred, X_collocation_torch)

        dPdt = grads[:, 1:2]
        dPdx = grads[:, 0:1]    
        dPdx2 = grad(dPdx, X_collocation_torch)[:, 0:1]
        
        phi = None
        k = None
        mu = None
        ct = None
        if model.parameter_discovery != None:
            index_parameter: int = 0
            for parameter in model.parameter_discovery:
                if parameter == "phi":
                    phi = model.unkown_parameters[index_parameter]
                if parameter == "k":
                    k = model.unkown_parameters[index_parameter]
                if parameter == "mu":
                    mu = model.unkown_parameters[index_parameter]
                if parameter == "ct":
                    ct = model.unkown_parameters[index_parameter]
                
                index_parameter += 1
        
        if phi == None: phi = model.Problem.phi
        if k == None: k = model.Problem.k
        if mu == None: mu = model.Problem.mu
        if ct == None: ct = model.Problem.ct

        pde = dPdt - (k / (mu * phi * ct)) * dPdx2

        return torch.mean(pde**2)

    #* Loss initial condition
    def loss_ic(model: PINN, numpoints=100):
        
        if model.parameter_discovery != None:
            index_parameter: int = 0
            for parameter in model.parameter_discovery:
                if parameter == "phi":
                    model.Problem.phi = float(model.unkown_parameters[index_parameter].detach().numpy())
                if parameter == "k":
                    model.Problem.k = float(model.unkown_parameters[index_parameter].detach().numpy())
                if parameter == "mu":
                    model.Problem.mu = float(model.unkown_parameters[index_parameter].detach().numpy())
                if parameter == "ct":
                    model.Problem.ct = float(model.unkown_parameters[index_parameter].detach().numpy())
                
                index_parameter += 1

        x = np.linspace(0, model.Problem.L, numpoints)
        X0 = np.stack([x, model.Problem.startTime*np.ones_like(x)], axis=1)
        X0_torch = model.np_to_th(X0).requires_grad_(True)

        P_init = np.array([model.Problem.AnalyticalSolution(xi, model.Problem.startTime) for xi in x])
        P_init = torch.tensor(P_init[:, None], dtype=torch.float32)  # shape (numpoints, 1)

        P_pred = model(X0_torch)
        return torch.mean((P_pred - P_init)**2)

    # set hyperparameters
    lr = 0.01
    epochs = 5000
    input_dim = 2
    output_dim = 1
    # hidden_layers = [50, 50, 50, 50]
    hidden_layers = [50, 50]

    LossPINNVec = []
    LossPINNVec.append(LossPINN(loss_bc_left, 100))
    LossPINNVec.append(LossPINN(loss_bc_right, 100))
    LossPINNVec.append(LossPINN(physics_loss, 1))
    LossPINNVec.append(LossPINN(loss_ic, 50))

    model = PINN(input_dim, hidden_layers, output_dim, nn.Tanh, epochs, nn.MSELoss(), lr, LossPINNVec, problem)
    # model = PINN(input_dim, hidden_layers, output_dim, nn.Tanh, epochs, None, lr, LossPINNVec, problem)
    # parameter_discovery is a list of a single string ["phi"], ["k"], ["mu"], ["ct"]
    # model = PINN(input_dim, hidden_layers, output_dim, nn.Tanh, epochs, None, lr, LossPINNVec, problem, parameter_discovery=["phi"])

    model.train_nn(t_xtpts, t_ypts)
    model.plot_loss()
    model.plot_prediction()

    testVec = np.array([[0.5, t] for t in np.linspace(problem.startTime, problem.endTime, 10)])
    predVec = model.predict(model.np_to_th(testVec)).detach().numpy()
    print(predVec)

    if model.parameter_discovery != None:
        index_parameter: int = 0
        for parameter in model.parameter_discovery:
            print(f"Learned porosity {parameter} = ", float(model.unkown_parameters[index_parameter].detach()))   
            index_parameter += 1

def ParameterDiscovery():
    problem = DarcyTransientFlow(L=1.0, PLeft=1.0, PRight=0.0, k=1e-12, mu=1e-3, phi=0.2, ct=1e-9, startTime=0.001, endTime=0.07)

    x_obs = torch.rand(40).view(-1,1) * problem.L
    t_obs = torch.rand(40).view(-1,1) * (problem.endTime - problem.startTime) + problem.startTime
    x_obs = x_obs.flatten()
    t_obs = t_obs.flatten()

    mesh = torch.meshgrid(x_obs, t_obs, indexing='ij')
    X_obs = torch.stack((mesh[0].flatten(), mesh[1].flatten()), dim=1)

    P_obs = np.array([problem.AnalyticalSolution(x_obs.numpy(), t.item()) for t in t_obs])
    P_obs_with_noise = P_obs + 0.04 * torch.randn_like(x_obs).numpy()

    x = np.linspace(0, problem.L, 100)

    fig, ax = plt.subplots()
    line, = ax.plot(x, problem.AnalyticalSolution(x, t_obs[0].item()), lw=2)
    points = ax.scatter(x_obs.numpy(), P_obs_with_noise[0], label='Data points', color='blue', s=10, alpha=0.5)
    ax.set_xlabel('Position (m)')

    fig.subplots_adjust(bottom=0.25)

    ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax = ax_time,
        label = 'Time (s)',
        valmin = problem.startTime,
        valmax = problem.endTime,
        valinit = problem.startTime,
        valstep=t_obs.detach().numpy()
    )

    def update(val):
        line.set_ydata(problem.AnalyticalSolution(x, time_slider.val))
        points.set_offsets(np.c_[x_obs.numpy(), P_obs_with_noise[np.where(t_obs.numpy() == time_slider.val)[0][0]]])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    fig.suptitle('Transient Darcy Flow in 1D Domain', y=0.95)
    plt.show()

    a = 1

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
    #main_nn_torch() # use to PINN validation
    ParameterDiscovery()
    #EquationTest()