"""
Author: Jamal Toutouh (toutouh@mit.edu)

basic-neuron.py contains the code to create a basic neuron that is able to simulate logic functions: AND and OR
by using step function.
"""

import numpy as np


def step(input, threshold=2):
    activation = 1 if input >= threshold else 0
    return activation


class BasicNeuron:
    """ It encapsulates a basic neuron that is constructed according to a list of weights, a bias value,
    and an activation function. """

    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def feedforward(self, inputs):
        """ It applies the feedforward of the function: it weights the imputs, add bias, and applies the activation
        function. """
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation(total)


# Running examples...

# AND function
weights = np.array([1, 1])  # w1 = 1, w2 = 1
bias = 0  # b = 0
andFunction = BasicNeuron(weights, bias, step)
input = np.array([1, 1])  # x1 = 0, x2 = 1
output = andFunction.feedforward(input)
print('AND({}) = {}'.format(input, output))

# OR function
weights = np.array([2, 2])  # w1 = 0, w2 = 1
bias = 0  # b = 0
orFunction = BasicNeuron(weights, bias, step)
input = np.array([2, 3])  # x1 = 2, x2 = 3
output = orFunction.feedforward(input)
print('OR({}) = {}'.format(input, output))
