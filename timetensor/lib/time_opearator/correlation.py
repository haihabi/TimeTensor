import numpy as np


def calculate_correlation(input_x, input_y, epsilon=0.0001):
    s_x = np.std(input_x)
    s_y = np.std(input_y)
    scale = (s_x * s_y) + epsilon
    return np.mean(np.multiply(input_x - np.mean(input_x), input_y - np.mean(input_y))) / scale


def cross_correlation(input_tensor_a, input_tensor_b, step_size, n_steps):
    result = []
    shift_vector = np.linspace(-step_size * n_steps, step_size * n_steps, 2 * n_steps + 1)
    n_points = []
    shift_vector_out = []
    for shift in shift_vector:
        _, index_a, index_b = np.intersect1d(input_tensor_a.time, input_tensor_b.time + shift,
                                             return_indices=True)
        if len(index_a) == 0: continue
        result.append(calculate_correlation(input_tensor_a.data[index_a], input_tensor_b.data[index_b]))
        n_points.append(len(index_a))
        shift_vector_out.append(shift)
    return result, shift_vector_out, n_points
