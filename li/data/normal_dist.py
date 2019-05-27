import numpy as np
import matplotlib.pyplot as plt


def generate_data(mu, sigma, path):
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()
    np.savetxt(path, s, delimiter=',')


def from_csv(path):
    return np.genfromtxt(path, delimiter=',')


def random(n):
    data_random = np.random.normal(loc=100, scale=25, size=n)
    data_sorted = np.sort(data_random)

    return data_random, data_sorted
