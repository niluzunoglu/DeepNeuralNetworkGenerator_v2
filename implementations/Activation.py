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


class Softmax(Activation):
    
    def __init__(self):
        self.output = None 

    def forward(self, x):
        if x.ndim == 1: 
            x_stable = x - np.max(x)
            exps = np.exp(x_stable)
            self.output = exps / np.sum(exps)
        elif x.ndim == 2:
            x_stable = x - np.max(x, axis=1, keepdims=True)
            exps = np.exp(x_stable)
            self.output = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError("Softmax girdisi 1D veya 2D olmalıdır.")
        return self.output

    def backward(self, grad_output):
        if self.output is None:
            raise ValueError("Softmax backward çağrılmadan önce forward çağrılmalıdır.")
        
        grad_input = np.zeros_like(self.output)
        for i in range(self.output.shape[0]):
            s_i = self.output[i, :] # (K,)
            g_i = grad_output[i, :] # (K,)
            grad_input_sample = s_i * (g_i - np.sum(g_i * s_i))
            grad_input[i, :] = grad_input_sample
            
        return grad_input