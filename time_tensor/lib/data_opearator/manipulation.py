import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.core.funtion_base import as_tensor
from time_tensor.core.common import check_same_time


def flatten(tt: TimeTensor):
    data = tt.data.copy()
    return as_tensor(np.reshape(data, [len(tt), -1]), tt.time)


def concat(tt_0: TimeTensor, tt_1: TimeTensor, axis=-1):
    check_same_time(tt_0, tt_1)  # check that input have the same time
    if axis >= 0: axis = axis + 1  # shift axis by because of the time domain
    return TimeTensor(np.concatenate((tt_0.data, tt_1.data), axis=axis), tt_0.time)


def reshape(tt_0: TimeTensor, shape: list) -> TimeTensor:
    shape = np.asarray(shape)
    shape = np.insert(shape, 0, len(tt_0))
    return TimeTensor(data=np.reshape(tt_0.data, shape), time=tt_0.time)


def stack(tt_0: TimeTensor, tt_1: TimeTensor, axis=-1):
    check_same_time(tt_0, tt_1)  # check that input have the same time
    if axis >= 0: axis = axis + 1  # shift axis by because of the time domain
    return TimeTensor(np.stack((tt_0.data, tt_1.data), axis=axis), tt_0.time)
