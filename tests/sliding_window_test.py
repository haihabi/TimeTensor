import unittest
import time_tensor as tt
import numpy as np


class SlidingWindowTest(unittest.TestCase):
    def test_window_size(self):
        n = 1000
        data = np.linspace(0, 39, 40).reshape([10, 4])
        tensor = tt.as_tensor(data)

        def sliding_function(data):
            return np.max(data, axis=0).reshape(1, -1)

        def sliding_function_consecutive(data):
            return np.max(data, axis=0).reshape(1, -1)

        st = tt.time_functions.sliding_window(tensor, 2, sliding_function)
        st = tt.time_functions.sliding_window_consecutive(tensor, sliding_function_consecutive, 2)
        self.assertTrue(2 * len(st) == len(tensor))


if __name__ == '__main__':
    unittest.main()
