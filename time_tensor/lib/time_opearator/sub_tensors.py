import numpy as np
from time_tensor.core.tensor import TimeTensor


def split2sub_tensor(time_tensor, step_size, window_size, start_time: float = None, stop_time: float = None) -> list:
    '''
    :param time_tensor:
    :param step_size:
    :param start_time:
    :param stop_time:
    :param window_size:
    :return:
    '''
    if start_time is None: start_time = time_tensor.start_time()
    if stop_time is None: stop_time = time_tensor.end_time()
    low_time = np.linspace(start_time, stop_time - window_size,
                           np.ceil((stop_time - start_time) / step_size).astype('int'))
    high_time = np.linspace(start_time + window_size, stop_time,
                            np.ceil((stop_time - start_time) / step_size).astype('int'))
    data_vector = []
    for lt, ht in zip(low_time, high_time):  # loop over high and low time step
        data = time_tensor.data[(time_tensor.time >= lt) * (time_tensor.time < ht), :]
        time = time_tensor.time[(time_tensor.time >= lt) * (time_tensor.time < ht)]
        if len(data) > 0:
            data_vector.append(TimeTensor(data, time))

    return data_vector  # a list of sub time tensor
