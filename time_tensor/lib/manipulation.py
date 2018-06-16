import numpy as np
from time_tensor.core.tensor import TimeTensor


def time_unique(tt: TimeTensor) -> TimeTensor:
    time, index = np.unique(tt.time, return_index=True)
    data = tt.data[index]
    return TimeTensor(data, time)
