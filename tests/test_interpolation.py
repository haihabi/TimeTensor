import unittest
import time_tensor as tt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # self.assertEqual(True, False)
        n = 1000
        data = np.linspace(0, 39, 40).reshape([10, 4])
        time = np.concatenate([np.linspace(0, 3, 4), np.linspace(5, 10, 6)])
        tensor = tt.as_tensor(data, time)
        new_tt = tt.time_functions.time_interpolation(tensor)

    def test_something_2_step(self):
        # self.assertEqual(True, False)
        n = 1000
        data = np.linspace(0, 63, 64).reshape([16, 4])
        time = np.concatenate([np.linspace(0, 3, 4), np.linspace(5 + 2, 10 + 2, 6), np.linspace(20, 25, 6)])
        tensor = tt.as_tensor(data, time)
        new_tt = tt.time_functions.time_interpolation(tensor, 12)


if __name__ == '__main__':
    unittest.main()
