import unittest
import time_tensor as tt
import numpy as np


class IteratorTestCase(unittest.TestCase):
    def test_base_iterator(self):
        t0 = tt.as_tensor(np.linspace(0, 9, 10).reshape([-1, 1]))
        i = 0
        for d, t in t0:
            self.assertTrue(t == i)
            self.assertTrue(np.all(d == i))
            i += 1

    def test_base_iterator2d(self):
        data = np.concatenate([np.expand_dims(np.linspace(0, 8, 9).reshape([3, 3]), axis=0) for i in range(10)], axis=0)
        t0 = tt.as_tensor(data)
        i = 0
        for d, t in t0:
            self.assertTrue(t == i)
            self.assertTrue(np.all(d == i))
            i += 1


if __name__ == '__main__':
    unittest.main()
