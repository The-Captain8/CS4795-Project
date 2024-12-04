from NeuralNetwork import NeuralNetwork
import matplotlib.image as img

image_file = "./training/augmented_images/image2.jpg"
image = img.imread(image_file).flatten()

nn = NeuralNetwork(0.1, vector_dimension=9)
