import unittest
import time_tensor as tt
import numpy as np
from tests.common import generate_random_tensor, generate_time_tensor


class DataOpManipulationTest(unittest.TestCase):
    def test_flatten(self):
        n = 10
        tt0 = generate_random_tensor(n, [2, 3])
        self.assertTrue(len(tt0) == n)
        shape = tt0.shape()
        self.assertTrue(shape[0] == 2)
        self.assertTrue(shape[1] == 3)

        tt1 = tt.data_opearator.flatten(tt0)
        shape = tt0.shape()
        self.assertTrue(shape[0] == 2)
        self.assertTrue(shape[1] == 3)
        shape = tt1.shape()
        self.assertTrue(shape[0] == 6)
        self.assertTrue(len(shape) == 1)

    def test_time_alignment(self):
        n = 10
        tt0 = generate_random_tensor(n, [2, 3])
        n = 8
        tt1 = generate_random_tensor(n, [2, 3])
        tt0_a, tt1_a = tt.time_opearator.alignment(tt0, tt1)
        self.assertTrue(len(tt0_a) == len(tt1_a))
        self.assertFalse(len(tt0_a) == len(tt0))

    def test_split(self):
        n = 10
        tt0 = generate_random_tensor(n, [2, 3])
        tt_list = tt.time_opearator.split_single_step(tt0)
        self.assertTrue(len(tt_list) == 1)
        self.assertTrue(tt_list[0] == tt0)
        time_vector = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        tt1 = generate_time_tensor(time_vector, np.asarray([2, 1]))
        tt_list = tt.time_opearator.split_single_step(tt1)
        self.assertTrue(len(tt_list) == 2)
        self.assertTrue(np.sum([len(t) for t in tt_list]) == len(time_vector))


if __name__ == '__main__':
    unittest.main()
