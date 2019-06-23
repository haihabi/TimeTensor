import unittest
import numpy as np
import timetensor as tt


class MyTestCase(unittest.TestCase):
    def test_multiy_algiment(self):
        tt_list = []
        for i in range(10):
            t = np.unique(np.random.randint(0, 100, 250))
            d = np.reshape(np.random.rand(len(t)), [-1, 1])
            tt_list.append(tt.as_tensor(d, t))
        tt_list_new = tt.time_opearator.multiple_tesnor_alignment(tt_list)
        for tt_old, tt_new in zip(tt_list, tt_list_new):
            self.assertTrue(len(tt_old) >= len(tt_new))
            print("a")


if __name__ == '__main__':
    unittest.main()
