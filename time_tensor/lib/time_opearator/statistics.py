import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.lib import time_opearator


def mean(tt: TimeTensor):
    return np.mean(tt.data, axis=0)


def variance(tt: TimeTensor):
    tt_mean = mean(tt)
    tt_zero_mean = tt - tt_mean
    return np.mean(np.power(tt_zero_mean.data, 2), axis=0)


def standard(tt: TimeTensor):
    return np.sqrt(variance(tt))


#
#
# def correlation(tt: TimeTensor):
#     tt_flat = data_opearator.flatten(tt)
#     tt_shape = tt_flat.shape()
#     tt_mean = mean(tt)


def cross_correlation(tt_0: TimeTensor, tt_1: TimeTensor, epsilon: float = 0.001):
    if tt_0 == tt_1:
        return np.ones(tt_0.shape())
    if len(tt_0) != len(tt_1):
        tt_0, tt_1 = time_opearator.alignment(tt_0, tt_1)
    tt_0_mean = mean(tt_0)
    tt_1_mean = mean(tt_1)
    tt_0_std = np.sqrt(variance(tt_0) + epsilon)
    tt_1_std = np.sqrt(variance(tt_1) + epsilon)

    tt_0_zero_mean = tt_0 - tt_0_mean
    tt_1_zero_mean = tt_1 - tt_1_mean

    scale = np.multiply(tt_0_std, tt_1_std)
    return np.divide(np.mean(np.multiply(tt_0_zero_mean.data, tt_1_zero_mean.data), axis=0), scale)
