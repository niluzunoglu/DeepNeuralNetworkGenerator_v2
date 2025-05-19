# implementations/Network.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

class Network:

    def __init__(self):
        self.layers = []
        # Buraya loss fonksiyonu ekleicem.

    def add_layer(self, layer):
        self.layers.append(layer)
        logger.info(f"Katman eklendi: {layer}")

    def forward(self, input_data):

        logger.debug(f"Network forward pass başlıyor, input_data shape: {input_data.shape}")
        current_output = input_data

        for i, layer in enumerate(self.layers):
            logger.debug(f"  Layer {i+1} ({layer.name}) forward processing...")
            current_output = layer.forward(current_output)
            logger.debug(f"  Layer {i+1} ({layer.name}) output shape: {current_output.shape}")

        logger.debug("İleri yayılım tamamlandı.")
        return current_output

    def backward(self, loss_gradient, learning_rate):
        logger.debug("Network backward pass başlıyor...")
        current_gradient = loss_gradient
        for i, layer in reversed(list(enumerate(self.layers))):
            logger.debug(f"  Layer {i+1} ({layer.name}) backward processing...")
            current_gradient = layer.backward(current_gradient, learning_rate)
            logger.debug(f"  Layer {i+1} ({layer.name}) input_gradient shape: {current_gradient.shape}")
        logger.debug("Network backward pass tamamlandı. Katmanlar parametrelerini güncelliyor.")
        

    def predict(self, input_data):
        logger.info(f"Prediction for input shape: {input_data.shape}")
        return self.forward(input_data)

    def train(self, X_train, y_train, epochs, learning_rate, loss_function_instance, verbose_every_n_epochs=100):
        num_samples = X_train.shape[0]
        logger.info(f"Eğitim başlıyor: Epochs={epochs}, LR={learning_rate}, Örnek Sayısı={num_samples}, Loss Fonksiyonu={loss_function_instance}")

        history = {'loss': []} 

        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(num_samples):

                x_sample = X_train[i:i+1] 
                y_sample_true = y_train[i:i+1] 
                y_sample_pred = self.forward(x_sample)
                current_sample_loss = loss_function_instance.calculate(y_sample_true, y_sample_pred)
                epoch_loss += current_sample_loss
                loss_grad = loss_function_instance.backward(y_sample_true, y_sample_pred)
                self.backward(loss_grad, learning_rate)

            average_epoch_loss = epoch_loss / num_samples
            history['loss'].append(average_epoch_loss)

            if (epoch + 1) % verbose_every_n_epochs == 0 or epoch == 0 or epoch == epochs -1 :
                logger.info(f"Epoch {epoch+1}/{epochs} - Ortalama Kayıp: {average_epoch_loss:.6f}")

        logger.info("Eğitim tamamlandı.")
        return history

    def __str__(self):

        s = "Network Mimarisi:\n"
        if not self.layers:
            s += "  (Henüz katman eklenmemiş)\n"
        for i, layer in enumerate(self.layers):
            s += f"  Layer {i+1}: {layer}\n"
        return s