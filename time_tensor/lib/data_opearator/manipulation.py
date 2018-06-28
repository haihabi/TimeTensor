import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.core.funtion_base import as_tensor


def flatten(tt: TimeTensor):
    data = tt.data.copy()
    return as_tensor(np.reshape(data, [len(tt), -1]), tt.time)
