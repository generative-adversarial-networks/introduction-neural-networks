import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
import numpy as np
from neuron import Neuron, ActivationFunctions
from basic_neuron import BasicNeuron, StepActivationFunction
from basic_two_layer_neural_network import BasicNeuralNetwork
from two_layer_neural_network import TwoLayerNeuralNetwork



def test_basic_neuron():
    activation_function = StepActivationFunction

    # AND function
    print('Testing neuron tom implement AND function.')
    weights = np.array([1, 1])  # w1 = 1, w2 = 1
    bias = 0  # b = 0
    and_function = BasicNeuron(weights, bias, activation_function.step)
    print(and_function.show_configuration())
    input = np.array([1, 1])  # x1 = 0, x2 = 1
    output = and_function.feedforward(input)
    print('AND({}) = {}'.format(input, output))
    input = np.array([0, 1])  # x1 = 0, x2 = 1
    output = and_function.feedforward(input)
    print('AND({}) = {}'.format(input, output))
    print('')

    # OR function
    print('Testing neuron tom implement OR function.')
    weights = np.array([2, 2])  # w1 = 0, w2 = 1
    bias = 0  # b = 0
    or_function = BasicNeuron(weights, bias, activation_function.step)
    print(or_function.show_configuration())
    input = np.array([0, 0])  # x1 = 2, x2 = 3
    output = or_function.feedforward(input)
    print('OR({}) = {}'.format(input, output))
    input = np.array([1, 0])  # x1 = 2, x2 = 3
    output = or_function.feedforward(input)
    print('OR({}) = {}'.format(input, output))


def test_neuron():
    print('Testing a neuron...')
    activations = ActivationFunctions()
    weights = np.array([3, 1])  # w1 = 3, w2 = 1
    bias = -1  # b = -1
    n = Neuron(weights, bias, activations.sigmoid)
    x = np.array([4, 1])  # x1 = 4, x2 = 1
    print(n.show_configuration())
    print('Input: {}  --> Result: {}'.format(x, n.feedforward(x)))  # 0.9999938558253978


def test_basic_neural_network():
    print('Testing neural networks ...')
    activations = ActivationFunctions()
    network = BasicNeuralNetwork([0, 1], 0, activations.sigmoid)
    x = np.array([2, 3])
    print(network.show_configuration())
    print(network.feedforward(x))  # 0.7216325609518421
    print('Input: {}  --> Result: {}'.format(x, network.feedforward(x)))  # 0.9999938558253978


def test_neural_network():
    print('Testing general two-layer neural networks ...')
    bias = 1
    input_layer_size = 5
    hidden_layer_size = 4
    activation = ActivationFunctions().sigmoid
    network = TwoLayerNeuralNetwork(input_layer_size, hidden_layer_size, bias, activation)
    x = np.array([2, 3, 4, 5, 6])
    print(network.show_configuration())
    print('Input: {}  --> Result: {}'.format(x, network.feedforward(x)))  # 0.9999938558253978


test_basic_neuron()
print('')

test_neuron()
print('')

test_basic_neural_network()
print('')

test_neural_network()
print('')
