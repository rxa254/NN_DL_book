import random
import numpy as np
from tqdm import tqdm

# Activation Functions and Their Derivatives
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1.0 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1.0, 0.0)

# Mapping activation functions to their respective names
activation_functions = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'tanh': (tanh, tanh_prime),
    'relu': (relu, relu_prime)
}

class Network(object):

    def __init__(self, sizes, activations=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        # Set activation functions for each layer
        self.activations = []
        if activations is None:
            activations = ['sigmoid'] * (self.num_layers - 1)  # Default to sigmoid if not specified
        for act in activations:
            if act in activation_functions:
                self.activations.append(activation_functions[act])
            else:
                raise ValueError(f"Activation function '{act}' not supported.")

    def feedforward(self, a):
        for (b, w, (activation, _)) in zip(self.biases, self.weights, self.activations):
            a = activation(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        with tqdm(range(epochs), desc="Training Progress") as t:
            for j in t:
                random.shuffle(training_data)
                mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)

                if test_data:
                    evaluation_result = self.evaluate(test_data)
                    t.write(f"Epoch {j + 1}: {evaluation_result} / {n_test}")
                else:
                    t.set_postfix(epoch=f"{j + 1} complete", refresh=True)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for (b, w, (activation_func, _)) in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activation_func(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.activations[-1][1](zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activations[-l][1](z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
