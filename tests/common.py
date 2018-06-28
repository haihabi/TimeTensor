import numpy as np
import time_tensor as tt


def generate_random_tensor(n_steps: int, shape: list):
    t0 = tt.empty_tensor()
    for i in range(n_steps):
        t0.insert_sort(np.random.rand(*shape), i)
    return t0
