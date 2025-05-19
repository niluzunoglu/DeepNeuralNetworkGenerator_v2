# implementations/Loss.py

import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)
logger = logging.getLogger(__name__)

class Loss:
    def calculate(self, y_true, y_pred):
        raise NotImplementedError
    def backward(self, y_true, y_pred): 
        raise NotImplementedError
    def __str__(self):
        return self.__class__.__name__

class MeanSquaredError(Loss):

    def calculate(self, y_true, y_pred):
        """Ortalama Kare Hatası (MSE) hesaplar."""
        # y_true ve y_pred'in aynı boyutta olduğundan emin ol
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        """MSE kaybının y_pred'e göre gradyanını hesaplar."""
        # y_true ve y_pred'in aynı boyutta olduğundan emin ol
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        # Gradyan: 2 * (y_pred - y_true) / N
        # N = örnek sayısı (y_true.shape[0] veya toplam eleman sayısı)
        # Eğer batch olarak işliyorsak, gradyanı batch ortalaması almak daha doğru olabilir
        # Şimdilik her bir çıktı için gradyanı döndürelim
        return 2 * (y_pred - y_true) / y_true.size # veya y_true.shape[0]

class MeanAbsoluteError(Loss): 

    def calculate(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        # MAE gradyanı: sign(y_pred - y_true) / N
        return np.sign(y_pred - y_true) / y_true.size

class BinaryCrossEntropy(Loss):

    epsilon = 1e-15
    def calculate(self, y_true, y_pred):

        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss)

    def backward(self, y_true, y_pred):

        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        grad = (-(y_true / y_pred_clipped) + (1 - y_true) / (1 - y_pred_clipped)) / y_true.size
        return grad
    

class CategoricalCrossentropy(Loss):

    def __init__(self, epsilon=1e-15):
        super().__init__()
        self.epsilon = epsilon

    def calculate(self, y_true, y_pred):

        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        if not np.all((y_true == 0) | (y_true == 1)): 
             logger.warning("CategoricalCrossentropy: y_true one-hot encoded olmayabilir.")

        y_pred_clipped = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        loss_per_sample = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        mean_loss = np.mean(loss_per_sample)
        return mean_loss

    def backward(self, y_true, y_pred):

        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")

        y_pred_clipped = np.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        gradient = -y_true / y_pred_clipped
        return gradient / y_true.shape[0] 