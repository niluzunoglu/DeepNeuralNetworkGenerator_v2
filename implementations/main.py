import numpy as np
import logging
import sys
import os
from DenseLayer import DenseLayer
from Activation import Sigmoid, Linear
from Network import Network
from Loss import MeanSquaredError
from Activation import Tanh

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":

    print("\n NETWORK forward and backward ÖRNEĞİ")

    X_sample = np.array([[0.5, 0.2]])
    y_true_sample = np.array([[0.8]])
    learning_rate = 0.1

    print(f"Girdi Verisi (X_sample): {X_sample}")
    print(f"Hedef Çıktı (y_true_sample): {y_true_sample}")
    print(f"Öğrenme Oranı: {learning_rate}")

    # --- Ağ Mimarisi ve Elle Ayarlanmış Ağırlıklar ---
    # Katman 1: 2 giriş -> 2 nöron (Tanh aktivasyon)
    # Katman 2: 2 giriş -> 1 nöron (Sigmoid aktivasyon) - Çıktı katmanı

    nn = Network()
    print("\n--- Ağ Oluşturuluyor ---")

    # Katman 1
    layer1 = DenseLayer(input_size=2, output_size=2, activation_function=Tanh(), name="GizliKatman")

    layer1.weights = np.array([[0.1, 0.3],   # w11, w12 (input1'den h1,h2'ye)
                                [-0.2, 0.4]]) # w21, w22 (input2'den h1,h2'ye)
    layer1.biases = np.array([[0.05, -0.05]])
    nn.add_layer(layer1)
    print(f"Katman 1 Ağırlıkları:\n{layer1.weights}")
    print(f"Katman 1 Biasları:\n{layer1.biases}")

    # Katman 2 
    layer2 = DenseLayer(input_size=2, output_size=1, activation_function=Sigmoid(), name="CiktiKatmani")
    # Elle ağırlık ve bias ataması
    layer2.weights = np.array([[0.5],   # w1 (hidden1'den output1'e)
                                [-0.6]]) # w2 (hidden2'den output1'e)
    layer2.biases = np.array([[0.1]])
    nn.add_layer(layer2)
    print(f"Katman 2 Ağırlıkları:\n{layer2.weights}")
    print(f"Katman 2 Biasları:\n{layer2.biases}")

    print(f"\nOluşturulan Ağ Mimarisi:\n{nn}")

    print("Eğitim başlıyor ****** ")
    print("\n\n--- ADIM 1: İLERİ YAYILIM (Eğitim Öncesi) ---")

    l1_weights_before = layer1.weights.copy()
    l1_biases_before = layer1.biases.copy()
    l2_weights_before = layer2.weights.copy()
    l2_biases_before = layer2.biases.copy()

    y_pred_before_training = nn.forward(X_sample)
    print(f"Ağ Çıktısı (Tahmin Edilen Değer - Eğitim Öncesi): {y_pred_before_training}")
    # Elle hesaplama (yukarıdaki değerlerle):
    # Gizli Katman (layer1):
    # z1 = X_sample @ layer1.weights + layer1.biases
    #    = [[0.5, 0.2]] @ [[0.1, 0.3], [-0.2, 0.4]] + [[0.05, -0.05]]
    #    = [[0.5*0.1 + 0.2*(-0.2), 0.5*0.3 + 0.2*0.4]] + [[0.05, -0.05]]
    #    = [[0.05 - 0.04, 0.15 + 0.08]] + [[0.05, -0.05]]
    #    = [[0.01, 0.23]] + [[0.05, -0.05]] = [[0.06, 0.18]]
    # a1 = Tanh(z1) = Tanh([[0.06, 0.18]]) approx [[0.0599, 0.1781]]
    #
    # Çıktı Katmanı (layer2):
    # z2 = a1 @ layer2.weights + layer2.biases
    #    = [[0.0599, 0.1781]] @ [[0.5], [-0.6]] + [[0.1]]
    #    = [[0.0599*0.5 + 0.1781*(-0.6)]] + [[0.1]]
    #    = [[0.02995 - 0.10686]] + [[0.1]]
    #    = [[-0.07691]] + [[0.1]] = [[0.02309]]
    # a2 (y_pred) = Sigmoid(z2) = Sigmoid(0.02309) approx 0.50577
    print(f"(Beklenen yaklaşık değer: [[0.50577]])")

    # --- 2. Kayıp Hesaplama ---
    print("\n\n--- ADIM 2: KAYIP HESAPLAMA ---")
    loss_function = MeanSquaredError()
    initial_loss = loss_function.calculate(y_true_sample, y_pred_before_training)
    print(f"Kullanılan Kayıp Fonksiyonu: {loss_function}")
    print(f"Hesaplanan Kayıp (Eğitim Öncesi): {initial_loss:.6f}")

    # --- 3. Kayıp Gradyanını Hesaplama ---
    print("\n\n--- ADIM 3: KAYIP GRADYANI HESAPLAMA ---")
    loss_gradient = loss_function.backward(y_true_sample, y_pred_before_training)
    print(f"Kayıp Gradyanı (dL/dy_pred): {loss_gradient}")

    # --- 4. Geri Yayılım (Backward Pass) ---
    print("\n\n--- ADIM 4: GERİ YAYILIM ---")
    print(f"Öğrenme Oranı: {learning_rate}")
    nn.backward(loss_gradient, learning_rate)
    print("Geri yayılım tamamlandı. Ağırlıklar ve biaslar güncellendi.")

    print("\n--- Güncellenmiş Ağırlıklar ve Biaslar ---")
    print(f"Katman 1 Eski Ağırlıkları:\n{l1_weights_before}\nKatman 1 Yeni Ağırlıkları:\n{layer1.weights}")
    print(f"Katman 1 Eski Biasları:\n{l1_biases_before}\nKatman 1 Yeni Biasları:\n{layer1.biases}")
    print(f"Katman 2 Eski Ağırlıkları:\n{l2_weights_before}\nKatman 2 Yeni Ağırlıkları:\n{layer2.weights}")
    print(f"Katman 2 Eski Biasları:\n{l2_biases_before}\nKatman 2 Yeni Biasları:\n{layer2.biases}")

    # --- 5. Tekrar İleri Yayılım (Güncellenmiş Ağırlıklarla Kaybı Kontrol Et) ---
    print("\n\n--- ADIM 5: TEKRAR İLERİ YAYILIM (Eğitim Sonrası) ---")
    y_pred_after_training = nn.forward(X_sample)
    print(f"Ağ Çıktısı (Tahmin Edilen Değer - 1 Adım Eğitim Sonrası): {y_pred_after_training}")

    final_loss = loss_function.calculate(y_true_sample, y_pred_after_training)
    print(f"Hesaplanan Kayıp (1 Adım Eğitim Sonrası): {final_loss:.6f}")

    if final_loss < initial_loss:
        print("BAŞARILI: Bir eğitim adımı sonrası kayıp azaldı!")
    else:
        print("UYARI: Bir eğitim adımı sonrası kayıp azalmadı veya arttı. Geri yayılım implementasyonunu kontrol edin.")

    print("\n========== BASİT AĞ AKIŞ ÖRNEĞİ TAMAMLANDI ==========")