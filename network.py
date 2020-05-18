"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.
"""

# Libraries
import random

import numpy as np

from utils import sigmoid, sigmoid_prime
import matplotlib.pyplot as plt

# Core class


class Network:
    def __init__(self, sizes):
        """
        sizes: [2, 3, 1] -> 2 neurons in input layer, 3 neurons in hidden layer, 1 neuron in output layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        # random.seed(1)
        self.weights = [
            np.random.randn(j, i) for i, j in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.random.randn(j, 1) for j in self.sizes[1:]]

    def forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None, full_batch=False):
        if test_data:
            # https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/ea229ac6234b7f3373f351f0b18616ca47edb8a1/network.py#L62
            test_data = list(test_data)
            n_test = len(test_data)
            test_results = []

        # https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/ea229ac6234b7f3373f351f0b18616ca47edb8a1/network.py#L58
        training_data = list(training_data)
        n = len(training_data)
        for t in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:  # Update parameters
                if full_batch is False:  # mini-batch: a list of tuples, tuple: (x,y), x,y: arrays
                    self.update_mini_batch(mini_batch, learning_rate)
                else:
                    self.update_full_batch(mini_batch, learning_rate)   # Use full matrix of the mini-batch
            if test_data:
                test_result = self.evaluate(test_data)
                test_results.append(test_result)
                print("Epoch {}: {} / {}".format(t, test_result, n_test))
            else:
                print("Epoch {} complete".format(t))
        plt.plot(range(len(test_results)), test_results)
        i_max, r_max = test_results.index(max(test_results)), max(test_results)
        plt.scatter([i_max], [r_max], c='g')
        plt.text(i_max, r_max, '{}'.format(r_max))
        plt.show()

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        """
        # Store the sum of all samples' nablas
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            sample_nabla_b, sample_nabla_w = self.backprop(x, y)
            # Sum update
            nabla_b = [nb + snb for nb, snb in zip(nabla_b, sample_nabla_b)]
            nabla_w = [nw + snw for nw, snw in zip(nabla_w, sample_nabla_w)]
            self.weights = [w - learning_rate / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - learning_rate / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

    def update_full_batch(self, mini_batch, learning_rate):
        """Update the batch via training the full matrix of all training sample at the same time"""
        # Concatenate training samples
        # x: 784*1, full_x: 784*m | y: 10*1, full_y: 10*m
        full_x = mini_batch[0][0]
        full_y = mini_batch[0][1]
        for x, y in mini_batch[1:]:
            full_x = np.concatenate((full_x, x), axis=1)
            full_y = np.concatenate((full_y, y), axis=1)
        # Backpropagation
        nabla_b, nabla_w = self.backprop2(full_x, full_y)
        self.weights = [w - learning_rate / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

    # https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/ea229ac6234b7f3373f351f0b18616ca47edb8a1/network.py#L93

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop2(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + np.repeat(b, activation.shape[1], axis=1)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1).reshape([-1, 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape([-1, 1])
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # https://medium.com/@hindsellouk13/matrix-based-back-propagation-fe143ce2b2df
    def backprop3(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activation matrices, layer by layer
        zs = []  # list to store all the "sum of weighted inputs z" matrices, layer by layer
        i = 0
        for b, w in zip(self.biases, self.weights):
            # insert the vector of biases on the first column of the weight matrix
            w = np.insert(w, 0, b.transpose(), axis=1)
            i = i+1
            # insert ones on the first line of the matrix of activations
            activation = np.insert(activation, 0, np.ones(activation[0].shape), 0)
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = np.expand_dims(np.sum(delta, axis=1), axis=1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.expand_dims(np.sum(delta, axis=1), axis=1)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(pred == targ) for (pred, targ) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations-y)
