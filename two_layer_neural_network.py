"""
Author: Jamal Toutouh (toutouh@mit.edu)
two_layer_neural_network.py contains the code to create a neural network with:
    - input_layer_size inputs
    - a hidden layer with hidden_layer_size neurons
    - an output layer with 1 neuron (o1)
"""

from neuron import Neuron, ActivationFunctions
import numpy as np


class TwoLayerNeuralNetwork:
    """It encapsulates a two layer neural neuron with:
    - input_layer_size inputs
    - a hidden layer with hidden_layer_size neurons
    - an output layer with 1 neuron (o1)
    - the weights are randomly computed
    """

    def __init__(self, input_layer_size, hidden_layer_size, bias, activation):
        weights = np.array([0, 1])
        self.bias = bias
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation

        # # The Neuron class here is from the previous section
        self.hidden_layer = list()
        for i in range(self.hidden_layer_size):
            self.hidden_layer.append(Neuron(self.create_weights(self.input_layer_size), self.bias, self.activation))

        self.o1 = Neuron(self.create_weights(self.hidden_layer_size), self.bias, self.activation)

    def feedforward(self, x):
        if len(x) != self.input_layer_size:
            print('The input {} has not the size of the input of the network, which is {}'
                  .format(x,self.input_layer_size))
        hidden_layer_output = list()
        for neuron in self.hidden_layer:
          hidden_layer_output.append(neuron.feedforward(x))

        # The inputs for o1 are the outputs from h1 and h2
        output_o1 = self.o1.feedforward(np.array(hidden_layer_output))
        return output_o1

    def show_configuration(self):
        config_string = 'Two layer neural networ configuration: \n'
        config_string += '* Input layer size: {}\n'.format(self.input_layer_size)
        config_string += '* Hidden layer size: {}\n'.format(self.hidden_layer_size)
        for neuron in self.hidden_layer:

          config_string += '   - {}\n'.format(neuron.show_configuration())

        config_string +=' * Output layer size: {}\n'.format(1)
        config_string += '   - {}\n'.format(self.o1.show_configuration())
        return config_string

    def create_weights(self, size):
        return np.random.normal(size=size)


# # Test
# bias = 1
# input_layer_size = 5
# hidden_layer_size = 8
# activation = ActivationFunctions().sigmoid
# network = TwoLayerNeuralNetwork(input_layer_size, hidden_layer_size, bias, activation)
# x = np.array([2, 3, 4, 5, 6])
# print(network.show_configuration())
# print(network.feedforward(x))  # 0.8913968007125518
