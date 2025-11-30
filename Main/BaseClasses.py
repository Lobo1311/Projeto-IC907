from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import numpy as np

@dataclass
class BasicData(metaclass=ABCMeta):
    deactivate:bool = False

    def DeactivateAttr(self):
        self.deactivate = True

    def ActivateAttr(self):
        self.deactivate = False

    def __str__(self):
        fields = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"

    def __setattr__(self, name, value):
        if not self.deactivate:
            return super().__setattr__(name, value)
        else:
            if hasattr(self, name):
                return super().__setattr__(name, value)
            
            raise AttributeError(f"Cannot add new attribute '{name}' when object is deactivated.")

class Layer(BasicData, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.inputs:np.ndarray = None
        self.output:np.ndarray = None
        self.dinputs:np.ndarray = None

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Method must be implemented in subclass.")
    
    @abstractmethod
    def backward(self, dvalues):
        raise NotImplementedError("Method must be implemented in subclass.")

class Loss(Layer, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def calculate(self, output:np.ndarray, y:np.ndarray):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Optimizer(BasicData, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.learning_rate:float = -123456789.0
        self.decay_rate:float=0.0
        self.decay_step:int = 100000
        self.interation:int = 0
        self.momentum:float = 0.0

    @abstractmethod
    def update_params(self, layer):
        raise NotImplementedError("Method must be implemented in subclass.")
    
    @abstractmethod
    def pre_update_params(self):
        raise NotImplementedError("Method must be implemented in subclass.")
    
    @abstractmethod
    def post_update_params(self):
        raise NotImplementedError("Method must be implemented in subclass.")