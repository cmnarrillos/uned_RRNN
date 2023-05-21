import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                y_hat = self.predict(x_i)
                error = y[idx] - y_hat
                self.weights[1:] += self.learning_rate * error * x_i
                self.weights[0] += self.learning_rate * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        y_predicted = np.where(linear_output > 0, 1, 0)
        return y_predicted


class NeuralNetwork:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.layers = []

    def add_layer(self, n_neurons):
        self.layers.append(
            [Perceptron(learning_rate=self.learning_rate, n_iterations=self.n_iterations) for _ in range(n_neurons)])

    def fit(self, X, y):
        for layer in self.layers:
            layer_input = X
            for perceptron in layer:
                perceptron.fit(layer_input, y)
                layer_input = perceptron.predict(layer_input)
            y = layer_input

    def predict(self, X):
        layer_input = X
        for layer in self.layers:
            layer_output = []
            for perceptron in layer:
                layer_output.append(perceptron.predict(layer_input))
            layer_input = np.concatenate(layer_output, axis=1)
        return layer_input
