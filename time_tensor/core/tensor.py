import numpy as np
import copy
import pickle
from time_tensor.core.common import check_input_size


class TimeTensor(object):
    def __init__(self, data: np.ndarray = [], time: np.ndarray = []):
        # if not isinstance(data, np.ndarray): raise Exception('input data type must be numpy array')
        # if len(data.shape) == 1: data = np.reshape(data, [-1, 1])

        self.time = time
        self.data = data
        self.i = 0

    def index_slice(self, start_index, stop_index):
        return TimeTensor(self.data[start_index:stop_index, :], self.time[start_index:stop_index])

    def get_time(self, index: int) -> np.ndarray:
        return self.data[index, :]

    def is_sorted(self):
        return np.array_equal(np.sort(self.time), self.time)

    def min_step(self) -> float:
        return np.min(np.diff(np.sort(self.time)))

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

    def shape(self) -> list:
        """
        The function return the data shape.
        :return: list - The data shape
        """
        return self.data.shape[1:]

    def insert_sort(self, data: np.ndarray, time: float):
        """
        The function insert a data array of 1d in the location according to the time value.
        :param data: ndarray - a 1d array of the data vector
        :param time: float - the data vector time value
        :return: TimeTensor - return the current instance of the TimeTensor
        """
        data = np.expand_dims(np.asarray(data), axis=0)  # make sure that the data is ndarray
        if len(self) > 0 and self.shape() != data.shape[
                                             1:]:  # check that dim size of the input must be the same as the current data
            raise ValueError
        i = np.searchsorted(self.time, time)  # search insertion index
        if len(self) == 0:
            self.data = data
        else:
            self.data = np.insert(self.data, i, data, axis=0)  # insert the data in the index location

        self.time = np.insert(self.time, i, time, axis=0)  # insert
        return self

    def insert(self, data: np.ndarray, time: float):
        np.append(self.data, data)
        np.append(self.time, time)
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
        if isinstance(other, TimeTensor):
            check_input_size(self, other)
            return TimeTensor(self.data + other.data, self.time)
        else:
            return TimeTensor(self.data + other, self.time)

    def __truediv__(self, other):
        if isinstance(other, TimeTensor):
            check_input_size(self, other)
            return TimeTensor(np.divide(self.data, other.data), self.time)
        else:
            return TimeTensor(self.data / other, self.time)

    def __mul__(self, other):
        if isinstance(other, TimeTensor):
            check_input_size(self, other)
            return TimeTensor(np.multiply(self.data, other.data), self.time)
        else:
            return TimeTensor(self.data * other, self.time)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, TimeTensor):
            check_input_size(self, other)
            return TimeTensor(self.data - other.data, self.time)
        else:
            return TimeTensor(self.data - other, self.time)

    def __getitem__(self, *args, **kwargs):
        if isinstance(args[0], TimeTensor):
            check_input_size(self, args[0])
            return TimeTensor(data=self.data[args[0].data], time=self.time)
        else:
            if isinstance(args[0], tuple):
                args = tuple([tuple([slice(None, None, None), *args[0]])])
            else:
                args = tuple([tuple([slice(None, None, None), args[0]])])
            new_data = self.data.__getitem__(*args, **kwargs)
            if len(new_data.shape) == 1:
                new_data = np.reshape(new_data, [-1, 1])
            return TimeTensor(data=new_data, time=self.time)

    def __len__(self) -> int:
        return len(self.time)

    def __copy__(self):
        return TimeTensor(self.data.copy(), self.time.copy())

    def __lt__(self, other):  # less then
        if isinstance(other, TimeTensor):
            raise NotImplemented
        else:
            return TimeTensor(data=self.data < other, time=self.time)

    def __le__(self, other):  # less equal
        if isinstance(other, TimeTensor):
            raise NotImplemented
        else:
            return TimeTensor(data=self.data <= other, time=self.time)

    def __gt__(self, other):  # greater then
        if isinstance(other, TimeTensor):
            raise NotImplemented
        else:
            return TimeTensor(data=self.data > other, time=self.time)

    def __ge__(self, other):
        if isinstance(other, TimeTensor):
            raise NotImplemented
        else:
            return TimeTensor(data=self.data >= other, time=self.time)

    def __setitem__(self, key, value):
        if isinstance(key, TimeTensor):
            check_input_size(self, key)
            self.data[key.data] = value
        else:
            if isinstance(key, tuple):
                if len(key) != len(self.shape()): raise IndexError
                key = tuple([slice(None, None, None), *key])
            else:
                if 1 != len(self.shape()): raise IndexError
                key = tuple([slice(None, None, None), key])
            self.data[key] = value

    def __eq__(self, obj):
        return isinstance(obj, TimeTensor) and np.array_equal(self.data, obj.data) and np.array_equal(self.time, obj.time)
