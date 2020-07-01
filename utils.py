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


def plot_results(results, start=None, end=None, show_max=False, merge=False):
    """Plot a list of data and highlight the top point.

    Arguments:
        results: dict 
        {label: list of data}
    """
    if merge:   # Merge into one plot
        plt.figure(1)
        for lab in results:
            result = results[lab]
            result = result[start:end]
            plt.plot([int(i+start) for i in range(len(result))], result, label=lab)
            if show_max:
                i_max, r_max = result.index(max(result))+start, max(result)
                plt.scatter([i_max], [r_max], c='g')
                plt.text(i_max, r_max, '{}'.format(r_max))
        plt.legend(loc='best')
    else:
        for i, lab in enumerate(results):
            result = results[lab]
            result = result[start:end]
            plt.figure(i)
            plt.plot([int(i+start) for i in range(len(result))], result, label=lab)
            if show_max:
                i_max, r_max = result.index(max(result))+start, max(result)
                plt.scatter([i_max], [r_max], c='g')
                plt.text(i_max, r_max, '{}'.format(r_max))
            plt.legend(loc='best')
    plt.show()


def make_percentage(num_list, total):
    r_list = []
    for num in num_list:
        r_list.append(round(num/total, 4))
    return r_list
