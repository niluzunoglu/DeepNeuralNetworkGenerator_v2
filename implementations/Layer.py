# implementations/Layer.py
import numpy as np

# Tüm katmanlar için bir base class oluşturuldu ortak fonksiyonları tanımlamak amacıyla. 
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    # Forward propagation fonksiyonu,girdi verisini alır ve outputu hesaplayacak.
    # Her bir katman kendi forward fonksiyonunu yazmalı (input sekli fark edeceğinden dolayı)
    def forward(self, input_data):
        raise NotImplementedError("Her katman kendi forward propagationını implemente etmelidir.")

    # Backward propagation fonksiyonu ise geri yayılımı gerçekleşirecek ve güncelleştirilmiş parametreleri
    # dönecek
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Her katman kendi backward fonksiyonunu implemente etmelidir.")

    def __str__(self):
        return self.__class__.__name__