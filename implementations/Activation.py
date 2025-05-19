# implementations/Activation.py

import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)

logger = logging.getLogger(__name__)

# Tüm aktivasyon fonksiyonları için base class. 
# Hepsinin forward ve backward fonksiyonları olmalı 
# ki gradyan için ayrıca tanımlanmasın
class Activation:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError
    def __str__(self):
        return self.__class__.__name__

class Sigmoid(Activation):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500))) 
        return self.output

    def backward(self, grad_output):
        # Sigmoid türevi: s(x) * (1 - s(x))
        sigmoid_derivative = self.output * (1 - self.output)
        return grad_output * sigmoid_derivative

class ReLU(Activation):
    def __init__(self):
        self.input_data = None

    def forward(self, x):
        self.input_data = x
        
        return np.maximum(0, x)

    def backward(self, grad_output):
        relu_derivative = np.where(self.input_data > 0, 1, 0)
        return grad_output * relu_derivative

class Tanh(Activation):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        tanh_derivative = 1 - (self.output**2)
        return grad_output * tanh_derivative

class Linear(Activation): 
     def forward(self, x):
        return x
     def backward(self, grad_output):
        return grad_output
