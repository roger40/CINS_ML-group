import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.array([i/1000 for i in range(1000)])
    plt.figure()
    plt.plot(np.tanh(x), label='tanh')
    plt.plot(np.exp(-x), label='exp')
    plt.legend()
    plt.show()