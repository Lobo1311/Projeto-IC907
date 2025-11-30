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

    # # implement each boundary or PINN loss funcs at this class following the struture of classes below 
    # def bc_loss_fn(model: nn.Module, x_train, y_pred: np.ndarray=None):
    #     # x0 = torch.tensor([[0.]], requires_grad=True)
    #     # v0 = model(x0)

    #     # dv0 = torch.autograd.grad(v0, x0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

    #     # bc_loss = v0.pow(2).mean() + dv0.pow(2).mean()

    #     bc_loss = 0.

    #     return bc_loss
    
    # def pinn_loss_fn(model: nn.Module, x_train: np.ndarray, y_pred: np.ndarray=None):
    #     # dv0 = torch.autograd.grad(y_pred, x_train, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    #     # dv1 = torch.autograd.grad(dv0, x_train, grad_outputs=torch.ones_like(dv0), create_graph=True)[0]

    #     # pinn_loss = (dv1 + P * (l - x_train) / (E*I)).pow(2).mean()

    #     pinn_loss = 0.

    #     return pinn_loss



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
    def __init__(self, input_dim:int, hidden_layers, output_dim:int, activation=nn.ReLU, epochs:int=5000, loss=nn.MSELoss(), lr:float=0.01, LossPINNVec:list[LossPINN]=[],
                 Problem:DarcyTransientFlow=None):
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
            # add activation after every hidden layer (not after final output layer)
            if i < len(sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers) # * is the unpacking operator
        self.loss_history = np.zeros(epochs)

        self.LossPINNVec: list[LossPINN] = LossPINNVec

        self.DeactivateAttr()

    def forward(self, x):
        return self.model(x)

    def predict(self, x) -> torch.Tensor:
        self.eval() # Puts the module and all its submodules into evaluation mode (sets their internal flag training = False).
        with torch.inference_mode(): # Disables autograd (no gradient computation) and also disables some autograd bookkeeping
            return self.forward(x)

    def np_to_th(self, x):
        """
        Convert a NumPy array to a PyTorch tensor.
        """
        n_samples = len(x)

        return torch.from_numpy(x).to(torch.float).reshape(n_samples,-1).requires_grad_(True) # reshape to (n_samples, input_dim) | -1 infers the second dimension
        # -1 also means “whatever dimension size is needed so that the total number of elements stays the same.”
        # requires_grad_(True) is important for PINNs since we need gradients w.r.t. inputs for computing derivatives

    def train_nn(self, x_train:np.ndarray=None, y_train:np.ndarray=None):  
        if x_train is not None and y_train is not None: 
            X = self.np_to_th(x_train)
            y = self.np_to_th(y_train)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            loss_value = 0.0

            if self.loss is not None:
                y_pred = self.forward(X)
                loss_value += self.loss(y_pred, y)

            for LossPINN in self.LossPINNVec:
                loss_value += LossPINN.LossFunc(self) * LossPINN.Weight #! check this

            loss_value.backward()

            optimizer.step()
            self.loss_history[epoch] = loss_value.item()
            if epoch % np.round(self.epochs/10) == 0 or epoch == self.epochs-1:
                lv = loss_value.item()
                entries = [("Total Loss", lv)]
                # if self.loss2 is not None:
                #     entries.append(("Loss2", loss2_value.item()))
                # if self.loss3 is not None:
                #     entries.append(("Loss3", loss3_value.item()))
                parts = []
                for name, val in entries:
                    parts.append(f"{name}: {val:.2e}" if val < 1e-7 else f"{name}: {val:.8f}")
                print(f"Epoch {epoch+1}/{self.epochs}, " + ", ".join(parts))

        #return loss_value.item()

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
        points = ax.scatter(x, y, label='Collocation points', color='blue', s=10, alpha=0.5)
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

        for t in [0.0, 0.025, 0.05, 0.075, 0.1]:
            print(t, float(self.predict(self.np_to_th(np.array([[0.5, t]])))))

        def update(val):
            line.set_ydata(self.Problem.AnalyticalSolution(x, time_slider.val))
            line2.set_ydata(self.predict(self.np_to_th(np.array([[xi, time_slider.val] for xi in x]))).detach().numpy())
            points.set_offsets(np.c_[x, self.Problem.AnalyticalSolution(x, time_slider.val)])
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        fig.legend()
        fig.suptitle('Transient Darcy Flow in 1D Domain', y=0.95)
        plt.show()