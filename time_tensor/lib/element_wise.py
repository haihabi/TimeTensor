import numpy as np
from time_tensor.core.tensor import TimeTensor
from time_tensor.core.common import check_input_size


def power(tt_0, tt_1) -> TimeTensor:
    if isinstance(tt_0, TimeTensor) and isinstance(tt_1, TimeTensor):
        check_input_size(tt_0, tt_1)
        return TimeTensor(data=np.power(tt_0.data, tt_1.data), time=tt_0.time)
    elif isinstance(tt_0, TimeTensor):
        return TimeTensor(data=np.power(tt_0.data, tt_1), time=tt_0.time)
    elif isinstance(tt_1, TimeTensor):
        return TimeTensor(data=np.power(tt_0, tt_1.data), time=tt_1.time)
    else:
        raise Exception('one of the input instance must be a TimeTensor')


def sqrt(tt_0) -> TimeTensor:
    return TimeTensor(np.sqrt(tt_0.data), tt_0.time)
