import unittest
import time_tensor as tt


class SaveLoadTest(unittest.TestCase):
    def test_something(self):
        t0 = tt.empty_tensor()
        t0.insert_sort([0, 1], 0)
        t0.insert_sort([0, 2], 1)
        t0.to_file('test')
        t1 = tt.from_file('test')
        self.assertTrue(t0 == t1)


if __name__ == '__main__':
    unittest.main()
