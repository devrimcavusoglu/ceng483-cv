from abc import abstractmethod

import numpy as np


class Operator:

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray, **kwargs):
        pass
