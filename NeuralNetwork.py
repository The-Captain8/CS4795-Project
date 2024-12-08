"""
Author: Nathan McGugan

Created with the tutorial at:
https://realpython.com/python-ai-neural-network/#adjusting-the-parameters-with-backpropagation

Modified to use ReLU activation function and some hidden neurons.
"""

import numpy as np


class NeuralNetwork:
    """
    A simple neural network with one hidden layer to predict a matrix from a flattened input vector.
    """

    def __init__(self, learning_rate, input_dimension, hidden_neurons=64, output_shape=(3, 3)):
        self.input_dimension = input_dimension
        self.hidden_neurons = hidden_neurons
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape) # For tic-tac-toe we always have a 3x3 output

        # Matrices are initially randomized
        # Input -> Hidden
        self.weights_input_to_hidden = np.random.randn(input_dimension, hidden_neurons) * np.sqrt(2.0 / input_dimension)
        self.bias_hidden = np.zeros(hidden_neurons)

        # Hidden -> Output
        self.weights_hidden_to_output = np.random.randn(hidden_neurons, self.output_size) * np.sqrt(2.0 / hidden_neurons)
        self.bias_output = np.zeros(self.output_size)

        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_prime(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_prime(self, x):
        return (x > 0).astype(float)

    def predict(self, input_vector):

        hidden_pre_activation = np.dot(input_vector, self.weights_input_to_hidden) + self.bias_hidden
        hidden_activation = self._relu(hidden_pre_activation)


        output_pre_activation = np.dot(hidden_activation, self.weights_hidden_to_output) + self.bias_output
        output_activation = self._sigmoid(output_pre_activation)

        return output_activation.reshape(self.output_shape), (hidden_pre_activation, hidden_activation, output_pre_activation)

    def _compute_gradients(self, input_vector, target_matrix):
        # Forward pass
        hidden_pre_activation = np.dot(input_vector, self.weights_input_to_hidden) + self.bias_hidden
        hidden_activation = self._relu(hidden_pre_activation)

        output_pre_activation = np.dot(hidden_activation, self.weights_hidden_to_output) + self.bias_output
        prediction = self._sigmoid(output_pre_activation)

        # Flatten target
        target_flat = np.array(target_matrix).flatten()

        # Compute error derivative wrt output
        d_error_d_output = 2 * (prediction - target_flat)
        
        # Derivatives at output layer
        d_output_pre_activation = self._sigmoid_prime(output_pre_activation) * d_error_d_output

        # Backprop from output to hidden
        d_error_d_weights_hidden_to_output = np.outer(hidden_activation, d_output_pre_activation)
        d_error_d_bias_output = d_output_pre_activation

        # Backprop to hidden layer
        d_hidden_activation = np.dot(d_output_pre_activation, self.weights_hidden_to_output.T)
        d_hidden_pre_activation = d_hidden_activation * self._relu_prime(hidden_pre_activation)

        d_error_d_weights_input_to_hidden = np.outer(input_vector, d_hidden_pre_activation)
        d_error_d_bias_hidden = d_hidden_pre_activation

        return (d_error_d_bias_hidden, d_error_d_weights_input_to_hidden,
                d_error_d_bias_output, d_error_d_weights_hidden_to_output)

    def _update_parameters(self, d_error_d_bias_hidden, d_error_d_weights_input_to_hidden,
                           d_error_d_bias_output, d_error_d_weights_hidden_to_output):
        self.bias_hidden -= self.learning_rate * d_error_d_bias_hidden
        self.weights_input_to_hidden -= self.learning_rate * d_error_d_weights_input_to_hidden
        self.bias_output -= self.learning_rate * d_error_d_bias_output
        self.weights_hidden_to_output -= self.learning_rate * d_error_d_weights_hidden_to_output

    def train(self, input_vectors, target_matrices, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target_matrix = target_matrices[random_data_index]

            gradients = self._compute_gradients(input_vector, target_matrix)
            self._update_parameters(*gradients)

            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target_matrix = target_matrices[data_instance_index]

                    prediction, _ = self.predict(data_point)
                    error = np.sum(np.square(prediction - target_matrix))
                    cumulative_error += error

                cumulative_errors.append(cumulative_error)

        return cumulative_errors
