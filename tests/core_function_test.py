import unittest
import time_tensor as tt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_insert(self):
        t0 = tt.empty_tensor()
        t0.insert([0, 1], 0)
        t0.insert([0, 2], 1)
        self.assertTrue(len(t0) == 2)
        self.assertTrue(t0.dim() == 2)

    def test_copy(self):
        t0 = tt.as_tensor(np.linspace(0, 9, 10).reshape([-1, 1]))
        t1 = t0.copy()
        self.assertFalse(t1 == t0)


if __name__ == '__main__':
    unittest.main()
