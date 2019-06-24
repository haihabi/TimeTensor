import numpy as np


def check_input_size(tt_0, tt_1):
    """
    Check input function

    :param tt_0:
    :param tt_1:
    :return:
    """
    check_same_time(tt_0, tt_1)
    if not np.array_equal(tt_0.shape(), tt_1.shape()):
        raise Exception('cant prefomane opeartion of time tensor with differant shapes')


def check_same_time(tt_0, tt_1):
    if not np.array_equal(tt_0.time, tt_1.time):
        raise Exception('cant prefomane opeartion of time tensor with differant time steps')
