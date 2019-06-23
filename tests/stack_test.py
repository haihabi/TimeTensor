import unittest
import numpy as np
import timetensor as tt


class MyTestCase(unittest.TestCase):
    def test_multiy_algiment(self):
        tt_list = []
        for i in range(10):
            t = np.unique(np.random.randint(0, 100, 250))
            t = t[:10]
            d = np.reshape(np.random.rand(len(t)), [-1, 1])
            tt_list.append(tt.as_tensor(d, t))
        tt.data_opearator.stack(tt_list, axis=0)


if __name__ == '__main__':
    unittest.main()
