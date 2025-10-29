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

    def _str_(self):
        fields = ", ".join(f"{key}={value}" for key, value in self._dict_.items())
        return f"{self._class.name_}({fields})"

    def _setattr_(self, name, value):
        if not self.deactivate:
            return super()._setattr_(name, value)
        else:
            if hasattr(self, name):
                return super()._setattr_(name, value)
            
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

    @abstractmethod
    def update_params(self, layer):
        raise NotImplementedError("Method must be implemented in subclass.")