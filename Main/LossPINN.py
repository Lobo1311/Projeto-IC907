import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LossPINN():
    def __init__(self):
        pass
    # implement each boundary or PINN loss funcs at this class following the struture of classes below 
    def bc_loss_fn(model: nn.Module, x_train, y_pred: np.ndarray=None):
        # x0 = torch.tensor([[0.]], requires_grad=True)
        # v0 = model(x0)

        # dv0 = torch.autograd.grad(v0, x0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        # bc_loss = v0.pow(2).mean() + dv0.pow(2).mean()

        bc_loss = 0.

        return bc_loss
    
    def pinn_loss_fn(model: nn.Module, x_train: np.ndarray, y_pred: np.ndarray=None):
        # dv0 = torch.autograd.grad(y_pred, x_train, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        # dv1 = torch.autograd.grad(dv0, x_train, grad_outputs=torch.ones_like(dv0), create_graph=True)[0]

        # pinn_loss = (dv1 + P * (l - x_train) / (E*I)).pow(2).mean()

        pinn_loss = 0.

        return pinn_loss