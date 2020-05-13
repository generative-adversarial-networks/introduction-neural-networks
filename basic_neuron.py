"""
Author: Jamal Toutouh (toutouh@mit.edu)

basic-neuron.py contains the code to create a basic neuron that is able to simulate logic functions: AND and OR
by using step function.
"""

import numpy as np



class StepActivationFunction:
    """ It encapsulates the step activation function. """

    def __init__(self):
        pass

    def step(input, threshold=2):
        activation = 1 if input >= threshold else 0
        return {'result': activation, 'name': 'step'}


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
        return self.activation(total)['result']

    def show_configuration(self):
        return 'Network configuration: ' \
               'Weights: {}, Bias: {}, and Activation: {}'.format(self.weights, self.bias,
                                                                  self.activation(0)['name'])

# # Running examples...
# activation_function = StepActivationFunction
#
# # AND function
# weights = np.array([1, 1])  # w1 = 1, w2 = 1
# bias = 0  # b = 0
# and_function = BasicNeuron(weights, bias, activation_function.step)
# input = np.array([1, 1])  # x1 = 0, x2 = 1
# output = and_function.feedforward(input)
# print('AND({}) = {}'.format(input, output))
#
# # OR function
# weights = np.array([2, 2])  # w1 = 0, w2 = 1
# bias = 0  # b = 0
# or_function = BasicNeuron(weights, bias, activation_function.step)
# input = np.array([2, 3])  # x1 = 2, x2 = 3
# output = or_function.feedforward(input)
# print('OR({}) = {}'.format(input, output))
