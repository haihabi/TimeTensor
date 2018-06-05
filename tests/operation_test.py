import unittest
import time_tensor as tt
import numpy as np


class TestBasicOperation(unittest.TestCase):
    def test_get_item(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        self.assertTrue(np.any(t0[0] == 0))

    def test_get_item_error(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        with self.assertRaises(Exception) as context:
            t0[1]
        self.assertTrue(isinstance(context.exception, IndexError))

    def test_add(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        t1 = t0 + 1
        self.assertTrue(t1.data[0, 0] > t0.data[0, 0])

    def test_div(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        t1 = t0 / 0.5
        self.assertTrue(t1.data[0, 1] > t0.data[0, 1])

    def test_mul(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        t1 = t0 * 2
        self.assertTrue(t1.data[0, 1] > t0.data[0, 1])

    def test_type_cast(self):
        t0 = tt.as_tensor(np.linspace(0, 10, 9).reshape([1, -1]))
        self.assertTrue(t0.data_type() == np.float64)
        t1 = t0.as_type(np.int8)  # change data type
        self.assertTrue(t0.data_type() == np.float64)
        self.assertTrue(t1.data_type() == np.int8)


if __name__ == '__main__':
    unittest.main()
