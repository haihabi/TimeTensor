import numpy as np
from timetensor.core.tensor import TimeTensor
from timetensor.core.funtion_base import as_tensor
from timetensor.core.common import check_same_time


def flatten(tt: TimeTensor) -> TimeTensor:
    """
    The flatten function a take a TimeTensor with arbratiry shape then return a TimeTensor with shape [N]
    where N is the number of element per step.

    :param tt: Input a TimeTensor object
    :return:  TimeTensor object
    """
    data = tt.data.copy()
    return as_tensor(np.reshape(data, [len(tt), -1]), tt.time)


def concat(tt_0: TimeTensor, tt_1: TimeTensor, axis: int = -1) -> TimeTensor:
    check_same_time(tt_0, tt_1)  # check that input have the same time
    if axis >= 0: axis = axis + 1  # shift axis by because of the time domain
    return TimeTensor(np.concatenate((tt_0.data, tt_1.data), axis=axis), tt_0.time)


def reshape(tt_0: TimeTensor, shape: list) -> TimeTensor:
    shape = np.asarray(shape)
    shape = np.insert(shape, 0, len(tt_0))
    return TimeTensor(data=np.reshape(tt_0.data, shape), time=tt_0.time)


def stack(*args, axis=-1) -> TimeTensor:
    """
    The stack function takes a set of TimeTensor and stack then on a given axis

    :param args: A list of TimeTensors
    :param axis: an int value that inditace the stack axis (-1 means add new axis at the end)
    :return: TimeTensor
    """
    if len(args) == 1:
        args = args[0]
    # check_same_time(tt_0, tt_1)  # check that input have the same time
    if axis >= 0: axis = axis + 1  # shift axis by because of the time domain
    new_data = np.stack([tt_c.data for tt_c in args], axis=axis)
    return TimeTensor(new_data, args[0].time)
