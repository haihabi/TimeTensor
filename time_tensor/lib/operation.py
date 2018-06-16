import numpy as np
from time_tensor.core.tensor import TimeTensor


def time_concat(tt_a: TimeTensor, tt_b: TimeTensor) -> TimeTensor:
    if len(tt_b) == 0:
        return tt_a
    elif len(tt_a) == 0:
        return tt_b
    return TimeTensor(data=np.concatenate((tt_a.data, tt_b.data), axis=0),
                      time=np.concatenate((tt_a.time, tt_b.time), axis=0))
