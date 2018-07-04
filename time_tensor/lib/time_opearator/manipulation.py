import numpy as np
from time_tensor.core.tensor import TimeTensor


def _build_alignment_vector(tt_0, tt_1):
    j = 0
    result = np.zeros((len(tt_0)), dtype=np.bool)
    result_b = np.zeros((len(tt_1)), dtype=np.bool)
    for i, t in enumerate(tt_0.time):
        if t == tt_1.time[j]:
            result[i] = True
            result_b[j] = True
            j += 1
        elif t > tt_1.time[j]:
            for _ in range(len(tt_1) - j):
                j += 1
                if t == tt_1.time[j]:
                    result[i] = True
                    result_b[j] = True
                elif t < tt_1.time[j]:
                    break
        if j >= len(tt_1): break
    return result, result_b


def time_unique(tt: TimeTensor) -> TimeTensor:
    time, index = np.unique(tt.time, return_index=True)
    data = tt.data[index]
    return TimeTensor(data, time)


def time_concat(tt_a: TimeTensor, tt_b: TimeTensor) -> TimeTensor:
    if len(tt_b) == 0:
        return tt_a
    elif len(tt_a) == 0:
        return tt_b
    return TimeTensor(data=np.concatenate((tt_a.data, tt_b.data), axis=0),
                      time=np.concatenate((tt_a.time, tt_b.time), axis=0))


def alignment(tt_0: TimeTensor, tt_1: TimeTensor, method='remove'):
    if method == 'remove':
        status_0, status_1 = _build_alignment_vector(tt_0, tt_1)
        return TimeTensor(tt_0.data[status_0, :], tt_0.time[status_0]), TimeTensor(tt_1.data[status_1, :],
                                                                                   tt_1.time[status_1])
    else:
        raise Exception('unknown alignment method')


def split(tt_0: TimeTensor, start=0, stop=-1):
    if start < 0: raise Exception('start index must be bigger the zero')
    if stop == -1: stop = len(tt_0)
    if stop < 0: raise Exception('stop index must be bigger the zero')
    return TimeTensor(data=tt_0.data[start:stop, :], time=tt_0.time[start:stop])


def split_single_step(tt_0: TimeTensor):
    if not tt_0.is_sorted():
        tt_0 = time_unique(tt_0)
    min_step = tt_0.min_step()
    diff_vec = np.diff(tt_0.time)
    split_index = np.where(diff_vec > min_step)[0]
    if len(split_index) == 0:
        return [tt_0]
    else:
        split_index = split_index + 1
        split_index = np.insert(split_index, 0, 0)
        split_index = np.append(split_index, -1)
        return [split(tt_0, start, stop) for start, stop in zip(split_index[:-1], split_index[1:])]
