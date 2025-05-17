from Layer import Layer
from DenseLayer import DenseLayer
from Activation import Sigmoid, ReLU, Tanh, Linear
from Loss import MeanSquaredError
import numpy as np

if __name__ == "__main__":

    input_size, output_size = 3, 2
    layer = DenseLayer(input_size, output_size, activation_function=Linear())
    layer.weights = np.array([[0.5], [0.2], [0.1]]) # (3,1)
    layer.biases = np.array([[0.01]])       # (1,1)

    input_data = np.array([[1.0, 2.0, 3.0]]) # (1,2)
    # Beklenen z = (1.0*0.5 + 2.0*0.2) + 0.1 = 0.5 + 0.4 + 0.1 = 1.0
    # Linear aktivasyon olduğu için output = z
    expected_output = np.array([[1.0]])

    output = layer.forward(input_data)

    print("Output:", output)
    print("Expected Output:", expected_output)
