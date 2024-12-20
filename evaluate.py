"""
Author: Nathan McGugan

A script to evaluate the performance of the neural network on the tic-tac-toe dataset.
"""

import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork
import cv2

DIMENSION = 256
ITERATIONS = 1000
HIDDEN_NEURONS = 128

input_vectors = []
target_matrices = []

annotations = pd.read_json("./training/augmented_annotations.json").to_dict()

def symbol_to_int(symbol):
    if symbol == "O":
        return 0
    elif symbol == "X":
        return 1
    else:
        return 2

def int_to_symbol(integer):
    if integer == 0 or integer < 0.5:
        return "O"
    elif integer == 1 or ( integer > 0.5 and integer < 1.5 ):
        return "X"
    else:
        return " "
    
def predict_and_print(network: NeuralNetwork, vector):
    predictions = network.predict(vector)[0]
    print(list(map(int_to_symbol, predictions[0])), '\n', \
          list(map(int_to_symbol, predictions[1])), '\n', \
          list(map(int_to_symbol, predictions[2])))
    
def evaluate(network: NeuralNetwork, input_vectors, target_matrices):
    """
    Returns the cumulative error and the average error
    """
    cumulative_error = 0
    error_values = []

    for i in range(len(input_vectors)):
        error_count = 0
        prediction = network.predict(input_vectors[i])[0]
        prediction = [int_to_symbol(element) for row in prediction for element in row]
        target_matrix = [int_to_symbol(element) for row in target_matrices[i] for element in row]

        for i in range(9):
            if prediction[i] != target_matrix[i]:
                error_count += 1
        
        error_rate = error_count / 9
        cumulative_error += error_rate
        error_values.append(error_rate)
    return cumulative_error / len(input_vectors), max(error_values), min(error_values)


for i in range(len(annotations["image"])):
    image_file = annotations["image"][i]
    image = cv2.imread(image_file)
    input_vector = cv2.resize(image, (DIMENSION, DIMENSION)).flatten() / 255.0
    input_vectors.append(input_vector.tolist())

    target_matrix_string = annotations["conversations"][i][1]['value']
    rows = target_matrix_string.split("\n")
    target_matrix = [[], [], []]

    for i, row in enumerate(rows):
        row_list = list(row)
        target_matrix[i].append(symbol_to_int(row_list[0]))
        target_matrix[i].append(symbol_to_int(row_list[1]))
        target_matrix[i].append(symbol_to_int(row_list[2]))
    target_matrices.append(target_matrix)


nn = NeuralNetwork(learning_rate=0.01, input_dimension=DIMENSION*DIMENSION*3, hidden_neurons=HIDDEN_NEURONS, output_shape=(3, 3))
errors = nn.train(input_vectors, target_matrices, iterations=ITERATIONS)

error = evaluate(nn, input_vectors, target_matrices)

print(f'Training iterations: {ITERATIONS}, Image dimension: {DIMENSION}X{DIMENSION}')
print(f'Average error: {error[0]}, Max error: {error[1]}, Min error: {error[2]}')