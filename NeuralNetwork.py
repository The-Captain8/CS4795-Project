"""
Author: Nathan McGugan

Created with the tutorial at:
https://realpython.com/python-ai-neural-network/#adjusting-the-parameters-with-backpropagation
"""

import numpy as np


class NeuralNetwork:
    """
    A simple neural network that can predict a single float from a multidimensional vector.
    """

    def __init__(self, learning_rate, vector_dimension=2):
        self.weights = np.array([np.random.randn() for _ in range(vector_dimension)])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)  # output is 'Prediction'
        return layer_2, layer_1

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        """
        In this whole 'derivative block of code' what we really want is the derivative of the error, with respect to 
        the weights. This tells us how the error changes when the weights are changed. It's hard to get an exact 
        derivative for that, so we get a couple derivatives that we can and multiply them together.

        This is the formula we will use:
        d_error/d_weights = d_error/d_prediction * d_prediction/d_layer1 * d_layer1/d_weights

        When multiplying these derivatives, a couple of the variables cancel out.
        This leaves us with the derivative of the error with respect to the weights.
        """

        # Derivative of the error with respect to the weights
        d_error_d_prediction = 2 * (prediction - target)

        # Derivative of the prediction with respect to layer1
        d_prediction_d_layer1 = self._sigmoid_prime(layer_1)

        # The derivative of layer1 with respect to the weights simplifies to the input
        d_layer1_d_weights = input_vector

        # d_layer1_d_bias is pretty easy to find as it is 1, since everything else in layer1 'disappears' when derived
        d_layer1_d_bias = 1

        d_error_d_weights = d_error_d_prediction * d_prediction_d_layer1 * d_layer1_d_weights

        # Similar process for the bias, and d_layer1_d_bias is pretty easy to find as it is 1
        d_error_d_bias = d_error_d_prediction * d_prediction_d_layer1 * d_layer1_d_bias

        return d_error_d_bias, d_error_d_weights

    def _update_parameters(self, d_error_d_bias, d_error_d_weights):
        self.bias = self.bias - (self.learning_rate * d_error_d_bias)
        self.weights = self.weights - (self.learning_rate * d_error_d_weights)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a random piece of data
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            d_error_d_bias, d_error_d_weights = self._compute_gradients(input_vector, target)

            self._update_parameters(d_error_d_bias, d_error_d_weights)

            # Measure the error across all test data every 100 iterations
            if current_iteration % 100 == 0:
                cumulative_error = 0

                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)[0]
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error

                cumulative_errors.append(cumulative_error)

        return cumulative_errors