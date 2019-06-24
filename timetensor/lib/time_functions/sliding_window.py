import numpy as np
from timetensor.core.tensor import TimeTensor
from timetensor.core.function_base import empty_tensor
from timetensor.lib.element_wise import sqrt
from timetensor.lib.time_functions.manipulation import alignment


def sliding_window(time_tensor: TimeTensor, step_size: float, window_function, start_time: float = None,
                   stop_time: float = None, window_size: float = None, enable_no_data: bool = False) -> TimeTensor:
    '''
    This function run a sliding window function over the time tensor and
    return the result in a new time tensor, The window function input is a ndarray of size NxM
    where M is the dim of the current time tensor and N is the number of samples in the current calculation.
    The output of the window function is ndarray of size 1xK one sample over K new features.
    :param time_tensor:
    :param step_size:
    :param window_function:
    :param start_time:
    :param stop_time:
    :param window_size:
    :param enable_no_data:
    :return:
    '''
    if start_time is None: start_time = time_tensor.start_time()
    if stop_time is None: stop_time = time_tensor.end_time()
    if window_size is None: window_size = step_size
    low_time = np.linspace(start_time, stop_time - window_size,
                           np.ceil((stop_time - start_time) / step_size).astype('int'))
    high_time = np.linspace(start_time + window_size, stop_time,
                            np.ceil((stop_time - start_time) / step_size).astype('int'))
    time_vector = []
    data_vector = []
    for lt, ht in zip(low_time, high_time):  # loop over high and low time step
        data = time_tensor.data[(time_tensor.time >= lt) * (time_tensor.time < ht), :]
        if len(data) > 0 or enable_no_data:
            data_new = window_function(data)
            if data_new is not None:
                data_vector.append(data_new)
                time_vector.append(ht)  # append time step
    if len(data_vector) == 0: return empty_tensor()
    return TimeTensor(np.concatenate(data_vector, axis=0),
                      np.asarray(time_vector))  # return a new time tensor after the sliding window


def moving_mean(tt_0, window_size, index=0):
    window = (1 / window_size) * np.ones(window_size)
    data = np.convolve(tt_0.data[:, index], window, mode='same')
    return TimeTensor(np.reshape(data, [-1, 1]), tt_0.time)


def moving_second_moment(tt_0, window_size, index=0):
    window = (1 / window_size) * np.ones(window_size)
    data = np.convolve(np.power(tt_0.data[:, index], 2), window, mode='same')
    return TimeTensor(np.reshape(data, [-1, 1]), tt_0.time)


def moving_cross_correlation(tt_0, tt_1, window_size, epsilon=0.0001):
    tt_0, tt_1 = alignment(tt_0, tt_1)
    if len(tt_0) == 0 or len(tt_1) == 0: return None
    mean_0 = moving_mean(tt_0, window_size)
    mean_1 = moving_mean(tt_1, window_size)
    xcorr_t = moving_mean((tt_0 - mean_0) * (tt_1 - mean_1), window_size)
    scale = sqrt(moving_second_moment(tt_0 - mean_0, window_size)) * sqrt(
        moving_second_moment(tt_1 - mean_1, window_size))
    return xcorr_t / (scale + epsilon)
