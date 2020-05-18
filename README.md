# Introduction to neural networks

This repository includes a series of *python* codes to introduce the user to the main concepts about **Neural Networks**.
These examples are part of my webinar/presentation about this topic. The presentation can be found here [**Introduction to neural networks**](https://jamaltoutouh.github.io/communication/introduction-neural-networks/).


### Code:

- ``neuron.py`` includes the code to create a basic neuron that uses different activation functions.

- ``basic-neuron.py`` contains the code to create a basic neuron that is able to simulate logic functions: AND and OR
by using step function.

- ``basic_two_layer_neural_network.py`` contains the code to create a basic neural network with:
  - two inputs: x1, and x2
  - a hidden layer with two neurons: h1 and h2
  - an output layer with a neuron: o1

- ``two_layer_neural_network.py`` contains the code to create a neural network with:
    - input_layer_size inputs
    - a hidden layer with hidden_layer_size neurons
    - an output layer with one neuron

- ``train-two-layer-neural-network.py`` contains the code to create and train a basic neural network with:
  - two inputs: x1, and x2
  - a hidden layer with two neurons: h1 and h2
  - an output layer with a neuron: o1



### Setup
The neural networks introduced here are implemented from scratch and using basic operations. There is no need for any kind of specific python packages, but *numpy* and *matplotlib*.

An easy way to install this dependency is using the environment file included in ``./auxfiles/`` folder, ``./auxfiles/nnenv.yml`` by using *conda*.

```
conda env create -f ./auxfiles/nnenv.yml
source activate nnenv
```

It could be done by running the following command and using the **nnenv** environemt (as it follows):
```
source activate nnenv
```

Or by installing  *numpy* and *matplotlib* using *pip* or *conda* in your system:

**conda**
```
conda install numpy matplotlib
```
**pip**
```
pip install numpy matplotlib
```



