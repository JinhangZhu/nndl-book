# Helper functions

# Libraries
import numpy as np

import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def plot_results(results):
    """Plot a list of data and highlight the top point.
    """
    plt.plot(range(len(results)), results)
    i_max, r_max = results.index(max(results)), max(results)
    plt.scatter([i_max], [r_max], c='g')
    plt.text(i_max, r_max, '{}'.format(r_max))
    plt.show()
