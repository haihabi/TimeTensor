import numpy as np
from time_tensor.core.tensor import TimeTensor


# from time_tensor.core.function_base import empty_tensor
# from time_tensor.lib.element_wise import sqrt
# from time_tensor.lib.time_functions.manipulation import alignment


def time_interpolation(tt: TimeTensor, max_fill: int = 6) -> TimeTensor:
    min_step_value = tt.min_step()
    c = np.round(np.diff(tt.time) / tt.min_step()).astype('int')
    index_list = np.where((c <= (max_fill + 1)) * (c > 1))[0]
    new_time = tt.time.copy()
    new_data = tt.data.copy()
    shift = 0
    for i in index_list:
        steps2add = (min_step_value) * np.linspace(1, c[i] - 1, c[i] - 1)
        data2add = tt.data[i, :].reshape(1, -1) * np.ones([c[i] - 1, 1])
        new_time = np.insert(new_time, i + shift, tt.time[i] + steps2add)
        new_data = np.insert(new_data, i + shift, data2add, axis=0)
        shift += (c[i] - 1)  # add values to shift
    return TimeTensor(new_data, new_time)

    # for i in index_list:
    #     size = c[i]
    #     index = i + shift
    #     time_v = tt.time[i] + (min_step_value) * np.linspace(1, c[i] - 1, c[i] - 1)
    #     data_v = tt.data[i, :].reshape(1, -1) * np.ones([size - 1, 1])
    #
    #     new_time = np.insert(new_time, index, time_v)
    #     new_data = np.insert(new_data, index, data_v, axis=0)
    #     shift += size - 1  # add values to shift
    # return TimeTensor(new_data, new_time)
