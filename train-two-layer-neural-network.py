"""
Author: Jamal Toutouh (toutouh@mit.edu)

train-two-layer-neural-network.py contains the code to create and train a basic neural network with:
- two inputs: x1, and x2
- a hidden layer with two neurons: h1 and h2
- an output layer with a neuron: o1
"""

import numpy as np
from neuron import ActivationFunctions
from networks_loss import NetoworksLoss
import matplotlib.pyplot as plt


def show_output(loss):
  y = np.array(loss)
  fig, ax = plt.subplots()
  ax.plot(y)
  ax.set(xlabel='training epochs (x10)', ylabel='loss',
         title='Loss evolution')
  ax.grid()
  plt.show()

class TrainedNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self, activation, activation_derivative, loss, learn_rate):
    self.activation = activation
    self.activation_derivative = activation_derivative
    self.loss = loss
    self.learn_rate = learn_rate

    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    # h1 = self.activation(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    # h2 = self.activation(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    # o1 = self.activation(self.w5 * h1 + self.w6 * h2 + self.b3)

    # --- Do a feedforward (we'll need these values later)
    sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
    h1 = self.activation(sum_h1)

    sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
    h2 = self.activation(sum_h2)

    sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
    o1 = self.activation(sum_o1)

    return o1, sum_h1, sum_h2, sum_o1, h1, h2

  def train(self, data, all_y_trues, epochs=1000):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    - epochs: number of times to loop through the entire dataset
    '''
    show = True
    loss_to_show = []

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # First   -- Compute the output of the network (o1 = y_pred) and the partial sums and activations
        #         -- We implemented all this in the feedforward function
        o1, sum_h1, sum_h2, sum_o1, h1, h2 = self.feedforward(x)
        y_pred = o1

        # Second  -- Compute partial derivatives.
        #         -- p_L_p_ypred represents the partial derivative of L over derivative y_pred
        p_L_p_ypred = -2 * (y_true - y_pred)

        # -- Output neuron (o1)
        p_ypred_p_w5 = h1 * self.activation_derivative(sum_o1)
        p_ypred_p_w6 = h2 * self.activation_derivative(sum_o1)
        p_ypred_p_b3 = self.activation_derivative(sum_o1)

        p_ypred_p_h1 = self.w5 * self.activation_derivative(sum_o1)
        p_ypred_p_h2 = self.w6 * self.activation_derivative(sum_o1)

        # -- Hidden layer neuron (h1)
        p_h1_p_w1 = x[0] * self.activation_derivative(sum_h1)
        p_h1_p_w2 = x[1] * self.activation_derivative(sum_h1)
        p_h1_p_b1 = self.activation_derivative(sum_h1)

        # -- Hidden layer neuron (h2)
        p_h2_p_w3 = x[0] * self.activation_derivative(sum_h2)
        p_h2_p_w4 = x[1] * self.activation_derivative(sum_h2)
        p_h2_p_b2 = self.activation_derivative(sum_h2)

        # Third   -- Update weights and biases
        # -- Hidden layer neuron (h1)
        self.w1 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_w1
        self.w2 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_w2
        self.b1 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_b1

        # -- Hidden layer neuron (h2)
        self.w3 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_w3
        self.w4 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_w4
        self.b2 -= self.learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_b2

        # -- Output neuron (o1)
        self.w5 -= self.learn_rate * p_L_p_ypred * p_ypred_p_w5
        self.w6 -= self.learn_rate * p_L_p_ypred * p_ypred_p_w6
        self.b3 -= self.learn_rate * p_L_p_ypred * p_ypred_p_b3

      # For logging purposes
      if epoch % 10 == 0:
        y_preds = [self.feedforward(dat)[0] for dat in data]
        loss = self.loss(all_y_trues, y_preds)
        loss_to_show.append(loss)
        print("Epoch %d loss: %.3f" % (epoch, loss))

    if show: show_output(loss_to_show)

# Define dataset
input = np.array([
  [-9, 0],  # Michael
  [11, -2],   # Shash
  [-10, -3],  # Hannah
  [8, 5],   # Lisa
])
real_y = np.array([
  0, # Michael
  1, # Shash
  0, # Hannah
  1, # Lisa
])


activation = ActivationFunctions.sigmoid
activation_derivative = ActivationFunctions.derivative_sigmoid
loss = NetoworksLoss.mse_loss
learning_rate = 0.1

# Train our neural network
network = TrainedNeuralNetwork(activation, activation_derivative, loss, learning_rate)
network.train(input, real_y)

# Make some predictions
jamal = np.array([-7, -3]) # 14 hours/week, 5
mina = np.array([10, -6])  # 31 hours/week, 2 papers
print("Jamal: %.3f" % network.feedforward(jamal)[0]) # No Data Scientist 0.04
print("Mina: %.3f" % network.feedforward(mina)[0]) # Data Scientist 0.945
