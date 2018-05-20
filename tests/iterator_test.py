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


if __name__ == '__main__':
    unittest.main()
