import unittest
import time_tensor as tt
import numpy as np


class SlidingWindowTest(unittest.TestCase):
    def test_something(self):
        data = np.linspace(0, 9, 10).reshape([10, 1])
        tensor = tt.as_tensor(data)

        def sliding_function(data):
            return np.max(data, axis=0).reshape(1, -1)

        st = tensor.sliding_window(2, sliding_function)
        self.assertTrue(2 * len(st) == len(tensor))


if __name__ == '__main__':
    unittest.main()
