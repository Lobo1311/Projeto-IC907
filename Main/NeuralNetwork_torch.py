import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


"""
Convert a NumPy array to a PyTorch tensor.
"""
def np_to_th(x):
  n_samples = len(x)
  return torch.from_numpy(x).to(torch.float).reshape(n_samples,-1).requires_grad_(True) # reshape to (n_samples, input_dim) | -1 infers the second dimension
  # -1 also means “whatever dimension size is needed so that the total number of elements stays the same.”
  # requires_grad_(True) is important for PINNs since we need gradients w.r.t. inputs for computing derivatives

"""
Count the number of trainable parameters in a PyTorch model.
"""
def count_parameters(model: nn.Module) -> int:
  nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'The model has {nparam} trainable parameters.')
  return nparam

class ANN(nn.Module):
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
    def __init__(self, input_dim: int, hidden_layers, output_dim: int, activation=nn.ReLU, epochs: int=5000, loss=nn.MSELoss(), lr: float=0.01,
                loss2: callable=None, loss2_weight: float=0.0, loss3: callable=None, loss3_weight: float=0.0):
        super().__init__()
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss3 = loss3
        self.loss3_weight = loss3_weight
        self.lr = lr
        self.epochs = epochs
        sizes = [input_dim] + hidden_layers + [output_dim]
        layers = []
        for i in range(len(sizes) - 1): # if len(sizes) = 4, i goes from 0 to 2
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            # add activation after every hidden layer (not after final output layer)
            if i < len(sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers) # * is the unpacking operator
        self.loss_history = np.zeros(epochs)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval() # Puts the module and all its submodules into evaluation mode (sets their internal flag training = False).
        with torch.inference_mode(): # Disables autograd (no gradient computation) and also disables some autograd bookkeeping
            return self.forward(x)

    def train_nn(self, x_train, y_train):
        X = np_to_th(x_train)
        y = np_to_th(y_train)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(X)
            loss_value = self.loss(y_pred, y) # Assumes loss is always callable
            if self.loss2 is not None:
                loss2_value = self.loss2(self, X, y_pred)
                loss_value += loss2_value * self.loss2_weight
            if self.loss3 is not None:
                loss3_value = self.loss3(self, X, y_pred)
                loss_value += loss3_value * self.loss3_weight
            loss_value.backward()
            optimizer.step()
            self.loss_history[epoch] = loss_value.item()
            if epoch % np.round(self.epochs/10) == 0 or epoch == self.epochs-1:
                lv = loss_value.item()
                entries = [("Total Loss", lv)]
                if self.loss2 is not None:
                    entries.append(("Loss2", loss2_value.item()))
                if self.loss3 is not None:
                    entries.append(("Loss3", loss3_value.item()))
                parts = []
                for name, val in entries:
                    parts.append(f"{name}: {val:.2e}" if val < 1e-7 else f"{name}: {val:.8f}")
                print(f"Epoch {epoch+1}/{self.epochs}, " + ", ".join(parts))
        return loss_value.item()

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_prediction(self, x_limits, x_training_data, y_training_data, analitycal_func: callable):
        x_vals = np.linspace(x_limits[0], x_limits[1], 100)
        x_tensor = np_to_th(x_vals)
        y_preds = self.predict(x_tensor).detach().numpy()
        plt.figure(figsize=(8,4))
        plt.plot(x_vals, y_preds, label='NN Prediction', color='C2', linewidth=2)
        plt.plot(x_vals, analitycal_func(x_vals), label='Analytical curve', color='C0', linewidth=2, linestyle='dashed')
        plt.scatter(x_training_data, y_training_data, label='Training data', color='C1', s=25, zorder=5)
        plt.xlabel('x (m)')
        plt.ylabel('Deflection v(x) (m)')
        plt.title('Neural Network Prediction vs Analytical Solution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()