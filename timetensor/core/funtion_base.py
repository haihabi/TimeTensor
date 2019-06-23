import numpy as np
from timetensor.core.tensor import TimeTensor


def as_tensor(data: np.ndarray, time: np.ndarray = None, is_sort=False) -> TimeTensor:
    """
    This function receive a 2d data array and 1d time array, then return a TimeTensor
    :param data: ndarray - data 2d array the first dim is the time index and the second dim is the feature dim
    :param time: ndarray - time 1d array
    :param is_sort: bool - a flag is the input is sorted
    :return: TimeTensor - TimeTensor based on the input data
    """
    assert isinstance(data, np.ndarray)

    if time is None:  # time is None build time series of step 1 start with zero
        time = np.linspace(0, data.shape[0] - 1, data.shape[0])
    assert len(time) == data.shape[0]  # check that data and time size match
    if not is_sort:  # if true sorting operation of data and time.
        index = np.argsort(time)  # get sorted index
        data = data[index, :]  # sort data
        time = time[index]  # sort time
    return TimeTensor(data, time)


def empty_tensor() -> TimeTensor:
    """
    This function return an empty TimeTensor
    :return: TimeTensor - empty TimeTensor
    """
    return TimeTensor()



