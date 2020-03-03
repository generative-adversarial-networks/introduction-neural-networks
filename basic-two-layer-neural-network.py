"""
Author: Jamal Toutouh (toutouh@mit.edu)

basic-two-layer-neural-network.py contains the code to create a basic neural network with:
- two inputs: x1, and x2
- a hidden layer with two neurons: h1 and h2
- an output layer with a neuron: o1
"""

from neuron import Neuron, ActivationFunctions
import numpy as np

class BasicNeuralNetwork:
    """
    A neural network with:
    - two inputs: x1, and x2
    - a hidden layer with two neurons: h1 and h2
    - an output layer with a neuron: o1
    The three neurons use the same weights and bias
    """

    def __init__(self, weights, bias, activation):
        self.h1 = Neuron(weights, bias, activation)
        self.h2 = Neuron(weights, bias, activation)
        self.o1 = Neuron(weights, bias, activation)

    def feedforward(self, x):
        """First we compute the output of the first layer"""
        output_h1 = self.h1.feedforward(x)
        output_h2 = self.h2.feedforward(x)

        """The outputs of the hiden layer h1 and h2 are the input of the output layer"""
        output_o1 = self.o1.feedforward(np.array([output_h1, output_h2]))

        return output_o1


# # Test
# activations = ActivationFunctions
# network = BasicNeuralNetwork([0, 1], 0, activations.sigmoid)
# x = np.array([2, 3])
# print(network.feedforward(x))  # 0.7216325609518421
