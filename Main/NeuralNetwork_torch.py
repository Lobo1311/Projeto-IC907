import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from BaseClasses import BasicData
from DarcyTransientFlow import DarcyTransientFlow

class LossPINN(BasicData):
    def __init__(self, LossFunc:callable=None, Weight:float=1.0):
        super().__init__()

        self.LossFunc = LossFunc
        self.Weight = Weight

        self.DeactivateAttr()

class PINN(nn.Module, BasicData):
    """
    Feed-forward neural network with configurable hidden layers.

    Parameters
    - input_dim (int): number of input features (e.g., 1 for position x).
    - hidden_layers (iterable[int]): sizes of hidden layers, e.g. [32, 16].
    - output_dim (int): number of outputs (e.g., 1 for deflection v(x)).
    - activation (nn.Module): activation class used between hidden layers (default: nn.ReLU).
    - epochs (int): number of training epochs.
    - loss (callable): loss function (default: nn.MSELoss()).
    - lr (float): optimizer learning rate.

    Notes
    - Uses ReLU (or provided activation) after each hidden layer and a linear output layer.
    - The forward() implementation accepts 1D tensors (unsqueezes to shape (N, input_dim)).
    """
    def __init__(self, input_dim:int, hidden_layers, output_dim:int, activation=nn.ReLU, epochs:int=5000, loss=nn.MSELoss(), lr:float=0.01, 
                 LossPINNVec:list[LossPINN]=[], Problem:DarcyTransientFlow=None, parameter_discovery:list[str]=None):
        super().__init__()

        if Problem is None:
            raise ValueError("Problem instance must be provided for PINN.")

        self.Problem = Problem
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        sizes = [input_dim] + hidden_layers + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
        self.loss_history = np.zeros(epochs)

        self.LossPINNVec: list[LossPINN] = LossPINNVec

        self.parameter_discovery: list[str] = parameter_discovery
        self.unkown_parameters: list = list()
        if self.parameter_discovery != None:
            for parameter in self.parameter_discovery:
                if parameter == "phi":
                    self.unkown_parameters.append(torch.nn.Parameter(torch.tensor(self.Problem.phi, dtype=torch.float32), requires_grad=True))
                if parameter == "k":
                    self.unkown_parameters.append(torch.nn.Parameter(torch.tensor(self.Problem.k, dtype=torch.float32), requires_grad=True))
                if parameter == "mu":
                    self.unkown_parameters.append(torch.nn.Parameter(torch.tensor(self.Problem.mu, dtype=torch.float32), requires_grad=True))
                if parameter == "ct":
                    self.unkown_parameters.append(torch.nn.Parameter(torch.tensor(self.Problem.ct, dtype=torch.float32), requires_grad=True))

        self.DeactivateAttr()

    def forward(self, x):
        return self.model(x)

    def predict(self, x) -> torch.Tensor:
        self.eval() 
        with torch.inference_mode(): 
            return self.forward(x)

    def np_to_th(self, x):
        n_samples = len(x)
        return torch.from_numpy(x).to(torch.float).reshape(n_samples,-1).requires_grad_(True)

    def train_nn(self, x_train:np.ndarray=None, y_train:np.ndarray=None):  
        if x_train is not None and y_train is not None: 
            X = self.np_to_th(x_train)
            y = self.np_to_th(y_train)

        if self.parameter_discovery != None:
            optimizer = optim.Adam(list(self.model.parameters()) + self.unkown_parameters, lr=self.lr)
        else:
            optimizer = optim.Adam(list(self.model.parameters()), lr=self.lr)

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            loss_value = 0.0

            if self.loss is not None:
                y_pred = self.forward(X)
                loss_value += self.loss(y_pred, y)

            for LossPINN in self.LossPINNVec:
                loss_value += LossPINN.LossFunc(self) * LossPINN.Weight

            loss_value.backward()

            optimizer.step()
            self.loss_history[epoch] = loss_value.item()
            if epoch % np.round(self.epochs/20) == 0 or epoch == self.epochs-1:
                print(f"Epoch {epoch+1}/{self.epochs}, Total Loss: {loss_value.item():.8f}")

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_prediction(self, numpoints:int=100):
        x = np.linspace(0, self.Problem.L, numpoints)

        fig, ax = plt.subplots()
        y = self.Problem.AnalyticalSolution(x, self.Problem.startTime)
        line, = ax.plot(x, y, lw=2)
        line2, = ax.plot(x, self.predict(self.np_to_th(np.array([[xi, self.Problem.startTime] for xi in x]))).detach().numpy(), lw=2, color='orange')
        # points = ax.scatter(x, y, label='Collocation points', color='blue', s=10, alpha=0.5)
        ax.set_xlabel('Position (m)')

        fig.subplots_adjust(bottom=0.25)

        ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax = ax_time,
            label = 'Time (s)',
            valmin = self.Problem.startTime,
            valmax = self.Problem.endTime,
            valinit = self.Problem.startTime,
        )

        def update(val):
            line.set_ydata(self.Problem.AnalyticalSolution(x, time_slider.val))
            line2.set_ydata(self.predict(self.np_to_th(np.array([[xi, time_slider.val] for xi in x]))).detach().numpy())
            # points.set_offsets(np.c_[x, self.Problem.AnalyticalSolution(x, time_slider.val)])
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        fig.legend()
        fig.suptitle('Transient Darcy Flow in 1D Domain', y=0.95)
        plt.show()