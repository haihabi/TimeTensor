import numpy as np
import copy
import pickle


class TimeTensor(object):
    def __init__(self, data: np.ndarray = [], time: np.ndarray = []):
        self.time = time
        self.data = data
        self.i = 0

    def to_file(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def data_type(self):
        """
        This function return the data type of the time tensor
        :return: Return the current data type
        """
        return self.data.dtype

    def as_type(self, input_type):
        """
        The function create a new copy of the time tensor and cast other type.
        :param input_type: The cast of the data type
        :return: A copy of the current time tensor after casting to other type
        """
        return TimeTensor(data=self.data.astype(input_type), time=self.time)

    def start_time(self) -> float:
        """
        The start time function return the last time stem of the time tensor
        :return:  a float value of the start time
        """
        return None if len(self.data) == 0 else self.time[0]

    def end_time(self) -> float:
        """
        The end time function return the last time stem of the time tensor
        :return:  a float value of the end time
        """
        return None if len(self.data) == 0 else self.time[-1]

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
        return self.data.shape[1:]

    def insert(self, data: np.ndarray, time: float):
        """
        The function insert a data array of 1d in the location according to the time value.
        :param data: ndarray - a 1d array of the data vector
        :param time: float - the data vector time value
        :return: TimeTensor - return the current instance of the TimeTensor
        """
        data = np.expand_dims(np.asarray(data), axis=0)  # make sure that the data is ndarray
        if len(self) > 0 and self.dim() != data.shape[
                                           1:]:  # check that dim size of the input must be the same as the current data
            raise ValueError
        i = np.searchsorted(self.time, time)  # search insertion index
        if len(self) == 0:
            self.data = data
        else:
            self.data = np.insert(self.data, i, data, axis=0)  # insert the data in the index location

        self.time = np.insert(self.time, i, time, axis=0)  # insert
        return self

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
