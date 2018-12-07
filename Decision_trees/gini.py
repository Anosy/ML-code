import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    p = np.arange(0.001, 1, 0.001, dtype=np.float)
    print(p.shape)
    gini = 2 * p * (1 - p)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2
    error = 1 - np.max(np.vstack((p, 1 - p)), 0)
    plt.plot(p, h, 'b-', lw=2, label='Entropy')
    plt.plot(p, gini, 'g-', lw=2, label='Gini')
    plt.plot(p, error, 'y--', lw=2, label='Error')
    plt.grid(ls='dotted')
    plt.legend(loc='upper left')
    plt.show()
