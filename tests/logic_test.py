import unittest
import numpy as np
from tests.common import generate_random_tensor


class LogicTest(unittest.TestCase):
    def test_equal(self):
        tt0 = generate_random_tensor(10, [2, 3])
        self.assertTrue(tt0 == tt0)
        tt1 = tt0 == 0
        self.assertTrue(tt1.data_type() == bool)
        tt2 = tt0 < 100000
        self.assertTrue(np.all(tt2.data))
        tt2 = tt0 < -10000
        self.assertTrue(not np.all(tt2.data))


if __name__ == '__main__':
    unittest.main()
