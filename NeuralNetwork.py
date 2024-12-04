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

    def __init__(self, learning_rate, input_dimension, output_shape=(3, 3)):
        self.input_dimension = input_dimension
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape)  # Number of neurons in the output layer

        # Initialize weights and biases for the network
        self.weights = np.random.randn(input_dimension, self.output_size)
        self.bias = np.random.randn(self.output_size)
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        return layer_2.reshape(self.output_shape), layer_1

    def _compute_gradients(self, input_vector, target_matrix):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        
        """
        In this whole 'derivative block of code' what we really want is the derivative of the error, with respect to 
        the weights. This tells us how the error changes when the weights are changed. It's hard to get an exact 
        derivative for that, so we get a couple derivatives that we can and multiply them together.

        When multiplying these derivatives, a couple of the variables cancel out.
        This leaves us with the derivative of the error with respect to the weights.
        """

        # Flatten the target matrix for gradient computation
        target_flat = np.array(target_matrix).flatten()
        input_vector = np.array(input_vector)

        # Derivative of the error with respect to the weights
        d_error_d_prediction = 2 * (prediction - target_flat)
        d_prediction_d_layer1 = self._sigmoid_prime(layer_1)
        d_layer1_d_weights = input_vector[:, np.newaxis]
        d_error_d_weights = np.dot(d_layer1_d_weights, (d_error_d_prediction * d_prediction_d_layer1)[np.newaxis, :])
        d_error_d_bias = d_error_d_prediction * d_prediction_d_layer1

        return d_error_d_bias, d_error_d_weights


    def _update_parameters(self, d_error_d_bias, d_error_d_weights):
        self.bias = self.bias - (self.learning_rate * d_error_d_bias)
        self.weights = self.weights - (self.learning_rate * d_error_d_weights)

    def train(self, input_vectors, target_matrices, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target_matrix = target_matrices[random_data_index]

            d_error_d_bias, d_error_d_weights = self._compute_gradients(input_vector, target_matrix)
            self._update_parameters(d_error_d_bias, d_error_d_weights)

            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target_matrix = target_matrices[data_instance_index]

                    prediction = self.predict(data_point)[0]
                    error = np.sum(np.square(prediction - target_matrix))
                    cumulative_error += error

                cumulative_errors.append(cumulative_error)

        return cumulative_errors
