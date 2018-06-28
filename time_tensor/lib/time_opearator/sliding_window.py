import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.core.funtion_base import empty_tensor


def sliding_window(time_tensor: TimeTensor, slide_step: float, window_function, start_time: float = None,
                   stop_time: float = None) -> TimeTensor:
    '''
    This function run a sliding window function over the time tensor and
    return the result in a new time tensor, The window function input is a ndarray of size NxM
    where M is the dim of the current time tensor and N is the number of samples in the current calculation.
    The output of the window function is ndarray of size 1xK one sample over K new features.
    :param time_tensor:
    :param slide_step:
    :param window_function:
    :param start_time:
    :param stop_time:
    :return:
    '''
    if start_time is None: start_time = time_tensor.start_time()
    if stop_time is None: stop_time = time_tensor.end_time()
    low_time = np.linspace(start_time, stop_time - slide_step,
                           np.ceil((stop_time - start_time) / slide_step).astype('int'))
    high_time = np.linspace(start_time + slide_step, stop_time,
                            np.ceil((stop_time - start_time) / slide_step).astype('int'))
    time_vector = []
    data_vector = []
    for lt, ht in zip(low_time, high_time):  # loop over high and low time step
        data = time_tensor.data[(time_tensor.time >= lt) * (time_tensor.time < ht), :]
        if len(data) > 0:
            data_vector.append(window_function(data))
            time_vector.append(ht)  # append time step
    if len(data_vector) == 0: return empty_tensor()
    return TimeTensor(np.concatenate(data_vector, axis=0),
                      np.asarray(time_vector))  # return a new time tensor after the sliding window
