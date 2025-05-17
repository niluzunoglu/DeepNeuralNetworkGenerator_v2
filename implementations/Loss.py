# implementations/Loss.py
import numpy as np


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

class MeanAbsoluteError(Loss): # Senin UI'da vardı
    def calculate(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match.")
        # MAE gradyanı: sign(y_pred - y_true) / N
        return np.sign(y_pred - y_true) / y_true.size
