"""
Author: Jamal Toutouh (toutouh@mit.edu)

neuron.py contains the code to create a basic neuron that uses different activation functions.
"""

import numpy as np


class ActivationFunctions:
    """ It groups some activation functions. """

    def __init__(self):
        pass

    def step(self, x, threshold):
        activation = 1 if x >= threshold else 0
        return {'result': activation, 'name': 'step'}

    def sign(self, x):
        activation = 1 if x >= 0 else -1
        return {'result': activation, 'name': 'sign'}

    def sigmoid(self, x):
        activation = 1 / (1 + np.exp(-x))
        return {'result': activation, 'name': 'sigmoid'}


class Neuron:
    """ It encapsulates a basic neuron that is constructed according to a list of weights, a bias value,
      and an activation function. """

    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def feedforward(self, inputs):
        """ It applies the feedforward of the function: it weights the inputs, add bias, and applies the activation
        function. """
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation(total)['result']

    def show_configuration(self):
        return 'Network configuration: ' \
               'Weights: {}, Bias: {}, and Activation: {}'.format(self.weights, self.bias,
                                                                  self.activation(0)['name'])


# # Test
# activations = ActivationFunctions()
# activation = activations.sigmoid
# weights = np.array([3, 1])  # w1 = 3, w2 = 1
# bias = -1  # b = -1
# n = Neuron(weights, bias, activation)
# x = np.array([4, 1])  # x1 = 4, x2 = 1
# print(n.feedforward(x))  # 0.9999938558253978
# print(n.show_configuration())
