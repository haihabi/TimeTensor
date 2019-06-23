from matplotlib import pyplot as plt
from timetensor.core.tensor import TimeTensor


def plot(tt: TimeTensor):
    plt.plot(tt.time, tt.data)
