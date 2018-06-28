import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.lib import time_opearator


def _build_axis_list(tt):
    ts = tt.shape()
    return [i + 1 for i in range(len(ts))]


def mean(tt: TimeTensor):
    return np.mean(tt.data, axis=_build_axis_list(tt))


def variance(tt: TimeTensor):
    tt_mean = mean(tt)
    tt_zero_mean = tt - tt_mean
    return np.mean(np.power(tt_zero_mean.data, 2), axis=_build_axis_list(tt))


def standard(tt: TimeTensor):
    return np.sqrt(variance(tt))
