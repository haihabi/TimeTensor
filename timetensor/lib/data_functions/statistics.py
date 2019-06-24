import numpy as np
from timetensor.core.tensor import TimeTensor


def _build_axis_list(tt):
    ts = tt.shape()
    return [i + 1 for i in range(len(ts))]


def mean(tt: TimeTensor) -> np.ndarray:
    return np.mean(tt.data, axis=_build_axis_list(tt))


def variance(tt: TimeTensor) -> np.ndarray:
    tt_mean = mean(tt)
    tt_zero_mean = tt - tt_mean
    return np.mean(np.power(tt_zero_mean.data, 2), axis=_build_axis_list(tt))


def standard(tt: TimeTensor) -> np.ndarray:
    return np.sqrt(variance(tt))
