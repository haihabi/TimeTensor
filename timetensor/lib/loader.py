import pickle
from timetensor.core.tensor import TimeTensor


def from_file(file_path) -> TimeTensor:
    with open(file_path, 'rb') as handle:
        tt = pickle.load(handle)
    if not isinstance(tt, TimeTensor):
        raise Exception('loaded a non TimeTensor object')
    return tt
