import numpy as np
import copy


class TimeTensor(object):
    def __init__(self, data: np.ndarray = [], time: np.ndarray = []):
        self.time = time
        self.data = data
        self.start_time = time[0]
        self.end_time = time[-1]
        self.i = 0

    def copy(self):
        """
        The function return a new copy of the current instance.
        :return: TimeTensor - a copy of the current TimeTensor
        """
        return copy.copy(self)

    def dim(self) -> int:
        """
        The function return the data dim.
        :return: int - The data dim
        """
        return self.data.shape[1]

    def insert(self, data: np.ndarray, time: float):
        """
        The function insert a data array of 1d in the location according to the time value.
        :param data: ndarray - a 1d array of the data vector
        :param time: float - the data vector time value
        :return: TimeTensor - return the current instance of the TimeTensor
        """
        data = np.asarray(data)  # make sure that the data is ndarray
        assert len(data.shape) == 1  # make sure that the data is 1d vector
        if len(self) > 0 and self.dim() != data.shape[
            0]: raise ValueError  # check that dim size of the input must be the same as the current data
        i = np.searchsorted(self.time, time)  # search insertion index
        self.data = np.insert(self.data, i, data, axis=0)  # insert the data in the index location
        if len(self) == 0: self.data = self.data.reshape([1, -1])  # reshape data array to 2d in the first iteration
        self.time = np.insert(self.time, i, time, axis=0)  # insert
        return self

    def sliding_window(self, slide_step: float, window_function, start_time: float = None,
                       stop_time: float = None):
        '''
        This function run a sliding window function over the time tensor and
        return the result in a new time tensor, The window function input is a ndarray of size NxM
        where M is the dim of the current time tensor and N is the number of samples in the current calculation.
        The output of the window function is ndarray of size 1xK one sample over K new features.
        :param slide_step:
        :param window_function:
        :param start_time:
        :param stop_time:
        :return:
        '''
        if start_time is None: start_time = self.start_time
        if stop_time is None: stop_time = self.end_time
        low_time = np.linspace(start_time, stop_time - slide_step,
                               np.ceil((stop_time - start_time) / slide_step).astype('int'))
        high_time = np.linspace(start_time + slide_step, stop_time,
                                np.ceil((stop_time - start_time) / slide_step).astype('int'))
        time_vector = []
        data_vector = []
        for lt, ht in zip(low_time, high_time):  # loop over high and low time step
            data_vector.append(window_function(self.data[(self.time >= lt) * (self.time < ht), :]))
            time_vector.append(ht)  # append time step
        return TimeTensor(np.concatenate(data_vector, axis=0),
                          time_vector)  # return a new time tensor after the sliding window

    def __iter__(self):
        """
        Returns itself as an iterator
        """
        return self

    def __next__(self):
        """
        Returns the next letter in the sequence or
        raises StopIteration
        """
        if self.i >= len(self):
            self.i = 0
            raise StopIteration
        t = self.time[self.i]
        d = self.data[self.i, :]
        self.i += 1
        return d, t

    def __add__(self, other):
        return TimeTensor(self.data + other, self.time)

    def __truediv__(self, other):
        return TimeTensor(self.data / other, self.time)

    def __mul__(self, other):
        return TimeTensor(self.data * other, self.time)

    def __sub__(self, other):
        return TimeTensor(self.data - other, self.time)

    def __getitem__(self, item):
        status = self.time == item
        if not any(status): raise IndexError
        return self.data[status, :]

    def __len__(self) -> int:
        return len(self.time)

    def __copy__(self):
        return TimeTensor(self.data.copy(), self.time.copy())
