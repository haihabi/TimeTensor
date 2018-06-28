import numpy as np
from time_tensor.core.tensor import TimeTensor


def mean_time(tt: TimeTensor):
    return np.mean(tt.data, axis=0)
