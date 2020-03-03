from neuron import Neuron, ActivationFunctions
import numpy as np


# ... code from previous section here


class TwoLayerNeuralNetwork:
    '''
  A neural network with:
    - input_layer_size inputs
    - a hidden layer with hidden_layer_size neurons
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''

    def __init__(self):
        weights = np.array([0, 1])
        self.bias = 0
        self.input_layer_size = 2
        self.hidden_layer_size = 5

        self.hidden_layer = list(Neuron)
        for i in range( self.hidden_layer_size):
            self.hidden_layer.append(Neuron(self.create_weights(self.input_layer_size), self.bias))

        # # The Neuron class here is from the previous section
        # self.h1 = Neuron(weights, bias)
        # self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(self.create_weights( self.hidden_layer_size), bias)


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

    def create_weights(self, size):
        return np.random.normal(size=size)


network = TwoLayerNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421
