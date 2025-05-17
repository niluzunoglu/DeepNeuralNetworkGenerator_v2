# implementations/DenseLayer.py
# 
import numpy as np
from Layer import Layer
#from Activation import sigmoid, relu 
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)

class DenseLayer(Layer):

    def __init__(self, input_size, output_size, activation_function=None, name="Dense"):

        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function 
        self.name = name

        # Weightler ve biaslar burada tanımlanıyor. Burayı kullanıcıya açmak gerekecek!
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size)) 

        self.input_data_shape = None # input_data'nın şeklini saklamak için
        self.z = None # Aktivasyondan önceki lineer çıktı (input @ weights + bias)

    def forward(self, input_data):

        # FLAG 1: Forward işlemi wx+b yapılıyor (mutlaka aktivasyon fonksiyonu tanımlanması gerek)
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases # Burada wx + b denklemini gerçekleştirdim.
        
        if self.activation_function:
            self.output = self.activation_function.forward(self.z)
            self.logger.info("Aktivasyon fonksiyonu seçildi: ", self.activation_function)

        else:
            self.logger.error("Aktivasyon fonksiyonu tanımlanmadı. Linear aktivasyon kullanılıyor.!!")

        return input_data # Şimdilik sadece girdiyi döndürsün

    def backward(self, output_gradient, learning_rate):


        # Bu metodu sen dolduracaksın.
        # İpucu:
        # 1. Eğer self.activation_function varsa:
        #    activation_gradient = self.activation_function.backward(output_gradient)
        # Else:
        #    activation_gradient = output_gradient
        # 2. weights_gradient = np.dot(self.input.T, activation_gradient)
        # 3. biases_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        # 4. input_gradient = np.dot(activation_gradient, self.weights.T)
        # 5. Ağırlıkları ve biasları güncelle:
        #    self.weights -= learning_rate * weights_gradient
        #    self.biases -= learning_rate * biases_gradient
        # 6. return input_gradient

        if self.activation_function:
            self.logger.info("Aktivasyon fonksiyonu seçildi: ", self.activation_function," devam ediliyor.")
            
            try:
                gradient = self.activation_function.backward(output_gradient)
            except:
                self.logger.error("Aktivasyon fonksiyonu geri yayılımı sırasında hata oluştu.")

        else:
            self.logger.error("UYARI ! : Aktivasyon fonksiyonu seçilmediğinden lineer aktivasyonla ilerlenecek!!")
            gradient = output_gradient

        # FLAG 2 : Backward burada yapılıyor!! Hata olursa buradan olma ihtimali var
        weights_gradient = np.dot(self.input.T, gradient)
        biases_gradient = np.sum(gradient, axis=0, keepdims=True)
        input_gradient = np.dot(gradient, self.weights.T)

        self.logger.info("Geri yayılım işlemi için gradyanlar hesaplandı.")
        self.logger.info("Ağırlıklar ve Biaslar güncelleniyor.")

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        self.logger.info("Katmanın weight ve bias değerleri güncellendi.")

        return input_gradient

    # Takibi kolaylaştırmak adına katmanın parametrelerini yazdırması için eklenmiştir.
    def show_current_parameters(self):
        print("---current parameters of the layer---")
        print("Weights : ",self.weights)
        print("Biases : ",self.biases)

    def __str__(self):
        act_name = self.activation_function.__class__.__name__ if self.activation_function else "Linear"
        return f"{self.name}({self.input_size} -> {self.output_size}, Activation: {act_name})"