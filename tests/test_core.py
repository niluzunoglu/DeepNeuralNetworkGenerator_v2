# test_core.py (ana proje dizininde veya bir 'tests' klasöründe olabilir)
import numpy as np

# --- implementations paketinden gerekli sınıfları import et ---
# Eğer implementations/__init__.py dosyasını düzenlediysen:
# from implementations import DenseLayer, Sigmoid, ReLU, Tanh, MeanSquaredError, Network (Network'ü sonra ekleyeceğiz)

from implementations.Layer import DenseLayer
from implementations.Activation import Sigmoid, ReLU, Tanh, Linear 
from implementations.Loss import MeanSquaredError, MeanAbsoluteError

def run_single_layer_test(input_size, output_size, activation_cls, input_data, expected_output_shape):
    """Tek bir katman için ileri ve geri yayılımı test eder."""
    print(f"\n--- Testing Single Layer: Dense({input_size} -> {output_size}, Activation: {activation_cls.__name__ if activation_cls else 'Linear'}) ---")

    activation_instance = activation_cls() if activation_cls else Linear() # Eğer None ise Linear kullan
    layer = DenseLayer(input_size, output_size, activation_function=activation_instance)
    print(layer)
    print("Initial Weights Shape:", layer.weights.shape)
    print("Initial Biases Shape:", layer.biases.shape)
    # print("Initial Weights:\n", layer.weights) # Çok büyük olabilir, isteğe bağlı

    print("\nInput Data (shape {}):\n".format(input_data.shape), input_data)

    # İleri Yayılım
    try:
        print("\n--- Forward Pass ---")
        layer_output = layer.forward(input_data.copy()) # Orijinal input'u değiştirmemek için kopya
        print("Layer Output (shape {}):\n".format(layer_output.shape), layer_output)
        assert layer_output.shape == expected_output_shape, \
            f"Hata: Çıktı boyutu beklenenden farklı! Beklenen: {expected_output_shape}, Gelen: {layer_output.shape}"
        print("Forward pass başarılı ve çıktı boyutu doğru.")
    except NotImplementedError:
        print("Hata: Layer.forward() metodu henüz implemente edilmemiş.")
        return
    except Exception as e:
        print(f"Forward pass sırasında bir hata oluştu: {e}")
        return

    # Geri Yayılım
    try:
        print("\n--- Backward Pass ---")
        # Dummy gradyan (bir sonraki katmandan veya kayıp fonksiyonundan geliyormuş gibi)
        # Çıktı ile aynı boyutta olmalı
        dummy_output_gradient = np.random.randn(*layer_output.shape)
        learning_rate = 0.01
        print("Dummy Output Gradient (shape {}):\n".format(dummy_output_gradient.shape), dummy_output_gradient)
        print(f"Learning Rate: {learning_rate}")

        # Ağırlıkların kopyasını al (güncelleme öncesi ve sonrası karşılaştırmak için)
        weights_before_update = layer.weights.copy()
        biases_before_update = layer.biases.copy()

        input_gradient = layer.backward(dummy_output_gradient, learning_rate)
        print("Input Gradient (shape {}):\n".format(input_gradient.shape), input_gradient)
        # Girdi gradyanının boyutu, katmanın girdisiyle aynı olmalı
        assert input_gradient.shape == input_data.shape, \
            f"Hata: Girdi gradyanı boyutu beklenenden farklı! Beklenen: {input_data.shape}, Gelen: {input_gradient.shape}"

        print("\nWeights (before update):\n", weights_before_update[0,0] if weights_before_update.size > 0 else "N/A") # Sadece ilk elemanı yazdır
        print("Weights (after update):\n", layer.weights[0,0] if layer.weights.size > 0 else "N/A")
        print("Biases (before update):\n", biases_before_update[0,0] if biases_before_update.size > 0 else "N/A")
        print("Biases (after update):\n", layer.biases[0,0] if layer.biases.size > 0 else "N/A")

        # Ağırlıkların ve biasların güncellendiğini kontrol et (çok küçük lr için fark az olabilir)
        if not np.allclose(weights_before_update, layer.weights) or \
           not np.allclose(biases_before_update, layer.biases):
            print("Backward pass başarılı, ağırlıklar ve/veya biaslar güncellendi.")
        else:
            print("Uyarı: Ağırlıklar ve biaslar backward pass sonrası değişmedi. (LR çok küçük veya implementasyon hatası olabilir)")

    except NotImplementedError:
        print("Hata: Layer.backward() metodu henüz implemente edilmemiş.")
    except Exception as e:
        print(f"Backward pass sırasında bir hata oluştu: {e}")

def run_loss_function_test(loss_cls, y_true, y_pred):
    """Tek bir kayıp fonksiyonunu ve gradyanını test eder."""
    print(f"\n--- Testing Loss Function: {loss_cls.__name__} ---")
    loss_instance = loss_cls()
    print(loss_instance)

    print("Y_true (shape {}):\n".format(y_true.shape), y_true)
    print("Y_pred (shape {}):\n".format(y_pred.shape), y_pred)

    try:
        loss_value = loss_instance.calculate(y_true, y_pred)
        print(f"Calculated Loss: {loss_value}")
        assert isinstance(loss_value, (float, np.float_)) or loss_value.ndim == 0, "Loss skaler olmalı"

        loss_gradient = loss_instance.backward(y_true, y_pred)
        print("Loss Gradient (shape {}):\n".format(loss_gradient.shape), loss_gradient)
        assert loss_gradient.shape == y_pred.shape, \
            f"Hata: Kayıp gradyanı boyutu y_pred ile aynı olmalı! Beklenen: {y_pred.shape}, Gelen: {loss_gradient.shape}"
        print("Loss function ve gradyanı başarıyla test edildi.")
    except NotImplementedError:
        print("Hata: Loss fonksiyonu veya backward metodu henüz implemente edilmemiş.")
    except Exception as e:
        print(f"Loss fonksiyonu testi sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    print("========== ÇEKİRDEK NN BİLEŞENLERİ TESTİ BAŞLADI ==========")

    # --- Test Verisi ve Parametreleri (Bunları değiştirebilirsin) ---
    num_samples = 5   # Test edilecek örnek sayısı
    input_features = 4 # Girdi katmanındaki nöron sayısı (özellik sayısı)

    # Katman Yapılandırması (Liste içinde (output_nöron_sayısı, aktivasyon_sınıfı) tuple'ları)
    # Örnek: 2 katmanlı bir ağ:
    #   - İlk katman: 3 nöron, ReLU aktivasyon
    #   - İkinci katman (çıktı katmanı): 2 nöron, Sigmoid aktivasyon (sınıflandırma için)
    # Şimdilik tek katmanları ayrı ayrı test edeceğiz.
    # "Network" sınıfını yazdığımızda bunu kullanacağız.
    # layer_configurations = [
    #     (3, ReLU),
    #     (2, Sigmoid)
    # ]

    # Tek Katman Testleri için
    hidden_layer_neurons = 3
    output_layer_neurons = 2 # Örneğin 2 sınıflı bir problem için

    # Rastgele girdi verisi oluştur (num_samples x input_features)
    X_test = np.random.randn(num_samples, input_features)

    # --- Tek Katman Testleri ---
    # Gizli katman testi (örneğin ReLU ile)
    run_single_layer_test(
        input_size=input_features,
        output_size=hidden_layer_neurons,
        activation_cls=ReLU,
        input_data=X_test,
        expected_output_shape=(num_samples, hidden_layer_neurons)
    )

    # Çıktı katmanı testi (örneğin Sigmoid ile, bir önceki katmanın çıktısını alacakmış gibi)
    # Bu test için input_size, bir önceki katmanın output_size'ı olur.
    run_single_layer_test(
        input_size=hidden_layer_neurons, # Bir önceki katmanın çıktısı
        output_size=output_layer_neurons,
        activation_cls=Sigmoid,
        input_data=np.random.randn(num_samples, hidden_layer_neurons), # Yeni dummy input
        expected_output_shape=(num_samples, output_layer_neurons)
    )

    # Lineer aktivasyonlu bir katman testi
    run_single_layer_test(
        input_size=input_features,
        output_size=1, # Regresyon için tek çıktı
        activation_cls=Linear, # Veya None da geçebilirdik
        input_data=X_test,
        expected_output_shape=(num_samples, 1)
    )

    # --- Kayıp Fonksiyonu Testleri ---
    # Rastgele y_true ve y_pred oluştur (çıktı katmanıyla uyumlu)
    y_true_test = np.random.randint(0, 2, size=(num_samples, output_layer_neurons)).astype(float) # Örneğin 0 veya 1
    # y_pred, genellikle bir aktivasyon fonksiyonundan (örn: Sigmoid) sonra 0-1 arasında olur
    y_pred_test = np.random.rand(num_samples, output_layer_neurons)

    run_loss_function_test(MeanSquaredError, y_true_test, y_pred_test)
    run_loss_function_test(MeanAbsoluteError, y_true_test, y_pred_test)


    # ---- İLERİDE: Network Sınıfı Testi (Aşama 1'de eklenecek) ----
    # print("\n\n========== NETWORK SINIFI TESTİ (HENÜZ AKTİF DEĞİL) ==========")
    # from implementations.network import Network # Bu sınıfı Aşama 1'de yazacağız

    # if 'Network' in locals() or 'Network' in globals(): # Eğer Network sınıfı varsa
    #     print("\n--- Ağ Oluşturma ve Eğitim Testi ---")
    #     # Örnek ağ yapılandırması (XOR problemi için)
    #     # Girdi: 2 özellik, Çıktı: 1 nöron (Sigmoid ile)
    #     xor_input_size = 2
    #     xor_output_size = 1

    #     nn = Network()
    #     nn.add_layer(DenseLayer(xor_input_size, 3, activation_function=Tanh())) # Gizli katman
    #     nn.add_layer(DenseLayer(3, xor_output_size, activation_function=Sigmoid())) # Çıktı katmanı

    #     print("Oluşturulan Ağ Mimarisi:")
    #     for i, layer in enumerate(nn.layers):
    #         print(f"Layer {i+1}: {layer}")

    #     # XOR veri seti
    #     X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    #     y_xor = np.array([[0], [1], [1], [0]]) # Çıktılar 2D olmalı (num_samples, output_features)

    #     loss_function = MeanSquaredError()
    #     learning_rate_nn = 0.1
    #     epochs_nn = 1000 # Daha fazla epoch gerekebilir

    #     print(f"\nEğitim Başlıyor (Epochs: {epochs_nn}, LR: {learning_rate_nn}, Loss: {loss_function})")
    #     # Network sınıfında train metodu implemente edildiğinde bu çalışacak
    #     # try:
    #     #     nn.train(X_xor, y_xor, epochs=epochs_nn, learning_rate=learning_rate_nn, loss_function=loss_function)
    #     #     print("\nEğitim Tamamlandı.")
    #     #     print("Eğitim Sonrası Tahminler (XOR):")
    #     #     for x_val, y_val_true in zip(X_xor, y_xor):
    #     #         prediction = nn.predict(x_val.reshape(1, -1)) # Tek örnek için predict
    #     #         print(f"Input: {x_val}, True: {y_val_true[0]}, Predicted: {prediction[0][0]:.4f} (Raw: {prediction[0][0]})")
    #     # except NotImplementedError:
    #     #     print("Hata: Network.train() veya Network.predict() henüz implemente edilmemiş.")
    #     # except Exception as e:
    #     #     print(f"Ağ eğitimi/tahmini sırasında bir hata oluştu: {e}")
    # else:
    #     print("Network sınıfı henüz tanımlanmamış, test atlanıyor.")


    print("\n========== ÇEKİRDEK NN BİLEŞENLERİ TESTİ TAMAMLANDI ==========")