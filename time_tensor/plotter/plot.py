from matplotlib import pyplot as plt
from time_tensor.core.tensor import TimeTensor


def plot(tt: TimeTensor):
    plt.plot(tt.time, tt.data)
