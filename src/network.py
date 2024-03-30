import numpy as np
import random
from tqdm import tqdm

class ActivationFunction:
    def __init__(self, name):
        self.name = name
        self.last_output = None  # Cache for storing the last output to avoid redundant calculations.

    def function(self, z):
        if self.name == 'sigmoid':
            z = np.clip(z, -500, 500)
            self.last_output = 1.0 / (1.0 + np.exp(-z))
        elif self.name == 'tanh':
            self.last_output = np.tanh(z)
        elif self.name == 'relu':
            self.last_output = np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation function: {self.name}")
        return self.last_output

    def derivative(self, z):
        if self.name == 'sigmoid':
            return self.last_output * (1 - self.last_output)
        elif self.name == 'tanh':
            return 1.0 - np.power(self.last_output, 2)
        elif self.name == 'relu':
            return np.where(z > 0, 1.0, 0.0)
        else:
            raise ValueError(f"Unsupported activation function derivative: {self.name}")

class Network:
    def __init__(self, sizes, activations=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # Initialize activation functions for each layer
        if activations is None:
            # Default to sigmoid if not specified
            activations = ['sigmoid'] * (self.num_layers - 1)
        elif len(activations) != (self.num_layers - 1):
            raise ValueError("The number of activation functions must match the number of layers minus one.")
        
        self.activations = [ActivationFunction(act) for act in activations]

    def feedforward(self, a):
        for b, w, activation in zip(self.biases, self.weights, self.activations):
            a = activation.function(np.dot(w, a) + b)
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
        for b, w, activation_func in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activation_func.function(z)
            activations.append(activation)

        # Output layer error
        delta = self.cost_derivative(activations[-1], y) * self.activations[-1].derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activations[-l].derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
