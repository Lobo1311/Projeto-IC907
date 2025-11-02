from BaseClasses import BasicData
import numpy as np
  
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