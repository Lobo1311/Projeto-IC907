from BaseClasses import BasicData
import numpy as np
import copy
  
class DataSet(BasicData):
    def __init__(self, x:np.ndarray, y:np.ndarray):
        super().__init__()
        self.x = x
        self.y = y

    def split(self, split_rate:float = 0.75):
        if split_rate < 0.0 or split_rate > 1.0:
            raise ValueError("Split rate must be between 0 and 1.")

        train_size = int(self.x.shape[0] * split_rate)
        
        perm = np.random.permutation(self.x.shape[0])
        
        x_train = self.x[perm][:train_size]
        y_train = self.y[perm][:train_size]

        x_test = self.x[perm][train_size:]
        y_test = self.y[perm][train_size:]

        train_set = DataSet(x_train, y_train)
        test_set = DataSet(x_test, y_test)

        return train_set, test_set
    
    def add_noise(self, randomization_factor:float=1.):
        """
        The random values is by default between -1 and 1.\n
        Use the randomization_factor to increase or decrease it.
        """
        ypts = copy.deepcopy(self.y)
        num_points: int = len(ypts)
        
        noise_points = (np.random.normal(-1., 1., num_points) * randomization_factor)
        y_randomized = ypts + noise_points.reshape(num_points, 1)
        
        self.y = copy.deepcopy(y_randomized)
        
        return 