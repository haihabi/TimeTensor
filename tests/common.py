import numpy as np
import timetensor as tt


def generate_random_tensor(n_steps: int, shape: list):
    t0 = tt.empty_tensor()
    for i in range(n_steps):
        t0.insert_sort(np.random.rand(*shape), i)
    return t0


def generate_time_tensor(time_list: list, data: np.ndarray):
    t0 = tt.empty_tensor()
    for i in time_list:
        t0.insert_sort(data, i)
    return t0
