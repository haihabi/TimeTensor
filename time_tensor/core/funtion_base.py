import numpy as np
from time_tensor.core.tensor import TimeTensor


def as_tensor(data: np.ndarray, time: np.ndarray = None) -> TimeTensor:
    """
    This function receive a 2d data array and 1d time array, then return a TimeTensor
    :param data: ndarray - data 2d array the first dim is the time index and the second dim is the feature dim
    :param time: ndarray - time 1d array
    :return: TimeTensor - TimeTensor based on the input data
    """
    assert isinstance(data, np.ndarray)
    if time is None:  # time is None build time series of step 1 start with zero
        time = np.linspace(0, data.shape[0] - 1, data.shape[0])
    assert len(time) == data.shape[0]  # check that data and time size match
    return TimeTensor(data, time)


def empty_tensor() -> TimeTensor:
    """
    This function return an empty TimeTensor
    :return: TimeTensor - empty TimeTensor
    """
    return TimeTensor()


def data_concat(tensor_a: TimeTensor, tensor_b: TimeTensor) -> TimeTensor:
    if len(tensor_a) != len(tensor_b):
        raise Exception('Tensor length must be the same')
    return TimeTensor(np.concatenate((tensor_a.data, tensor_b.data), axis=1), tensor_a.time)


def time_concat(tensor_a: TimeTensor, tensor_b: TimeTensor) -> TimeTensor:
    for d, t in tensor_b:
        tensor_a.insert(d, t)
    return tensor_a.copy()
