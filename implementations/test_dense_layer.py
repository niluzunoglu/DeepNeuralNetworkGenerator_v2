import unittest
import numpy as np
from DenseLayer import DenseLayer
from Activation import Sigmoid, Linear
from Network import Network
from Loss import MeanSquaredError
from Activation import Tanh

import logging

logging.basicConfig(    
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)
logger = logging.getLogger(__name__)

class TestDenseLayerForward(unittest.TestCase):

    def test_forward_linear_activation(self):
        
        input_size, output_size = 2, 1
        layer = DenseLayer(input_size, output_size, activation_function=Linear())
        layer.weights = np.array([[0.5], [0.2]]) # (2,1)
        layer.biases = np.array([[0.1]])       # (1,1)

        input_data = np.array([[1.0, 2.0]]) # (1,2)
        # Beklenen z = (1.0*0.5 + 2.0*0.2) + 0.1 = 0.5 + 0.4 + 0.1 = 1.0
        # Linear aktivasyon olduğu için output = z
        expected_output = np.array([[1.0]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                              err_msg="Linear forward pass sonucu yanlış.")
        self.assertEqual(output.shape, (1, output_size), "Çıktı boyutu yanlış.")

    def test_zero_weights_and_bias(self):
        input_size, output_size = 2, 1
        layer = DenseLayer(input_size, output_size, activation_function=Linear())
        layer.weights = np.array([[0.0], [0.0]])
        layer.biases = np.array([[0.0]])

        input_data = np.array([[3.0, -5.0]])
        expected_output = np.array([[0.0]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                            err_msg="Sıfır weight ve bias ile çıktı 0 olmalı.")
        self.assertEqual(output.shape, (1, output_size))

    def test_negative_weights(self):
        input_size, output_size = 2, 1
        layer = DenseLayer(input_size, output_size, activation_function=Linear())
        layer.weights = np.array([[-1.0], [2.0]])
        layer.biases = np.array([[0.5]])

        input_data = np.array([[4.0, 1.0]])  # z = (-1)*4 + 2*1 + 0.5 = -4 + 2 + 0.5 = -1.5
        expected_output = np.array([[-1.5]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                            err_msg="Negatif ağırlıklarla sonuç yanlış.")
        self.assertEqual(output.shape, (1, output_size))

    def test_multiple_inputs(self):
        input_size, output_size = 2, 1
        layer = DenseLayer(input_size, output_size, activation_function=Linear())
        layer.weights = np.array([[1.0], [1.0]])
        layer.biases = np.array([[0.0]])

        input_data = np.array([
            [1.0, 2.0],   # z = 1 + 2 = 3
            [0.0, 0.0],   # z = 0
            [-1.0, 3.0]   # z = -1 + 3 = 2
        ])
        expected_output = np.array([[3.0], [0.0], [2.0]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                            err_msg="Çoklu inputlarla forward pass sonucu yanlış.")
        self.assertEqual(output.shape, (3, output_size))

    def test_only_bias_effect(self):
        input_size, output_size = 2, 1
        layer = DenseLayer(input_size, output_size, activation_function=Linear())
        layer.weights = np.array([[0.0], [0.0]])
        layer.biases = np.array([[2.7]])

        input_data = np.array([[100.0, -200.0]])
        expected_output = np.array([[2.7]])  # sadece bias etkili

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                            err_msg="Sadece bias etkisi yanlış.")
        self.assertEqual(output.shape, (1, output_size))

    def test_forward_sigmoid_activation(self):
        input_size, output_size = 1, 1
        layer = DenseLayer(input_size, output_size, activation_function=Sigmoid())
        layer.weights = np.array([[0.0]]) # z = 0 olacak şekilde
        layer.biases = np.array([[0.0]])

        input_data = np.array([[10.0]]) # Girdi ne olursa olsun z = 0 olacak
        # Beklenen z = 0. Sigmoid(0) = 0.5
        expected_output = np.array([[0.5]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5,
                                              err_msg="Sigmoid forward pass sonucu yanlış.")

    def test_forward_multiple_samples_and_outputs(self):

        input_size, output_size = 3, 2
        num_samples = 4
        layer = DenseLayer(input_size, output_size, activation_function=Sigmoid())
        # Ağırlıklar ve biaslar rastgele kalsın, sadece şekil ve genel işleyişi test edelim
        input_data = np.random.randn(num_samples, input_size)
        output = layer.forward(input_data)

        self.assertEqual(output.shape, (num_samples, output_size),
                         "Birden fazla örnek ve çıktı için boyut yanlış.")
        # Sigmoid çıktısı 0-1 arasında olmalı
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1),
                        "Sigmoid çıktısı 0-1 aralığında olmalı.")

class TestCompleteNetworkFlow(unittest.TestCase):

    def setUp(self):
        logger.info(f"\n{self._testMethodName} için TestCompleteNetworkFlow.setUp() başlıyor.")
        self.input_data_sample = np.array([[0.5, -0.2]])
        self.expected_final_output_shape = (1, 1)
        self.learning_rate_test = 0.1
        self.loss_cls_for_test = MeanSquaredError

        layer1_weights = np.array([[0.1, 0.3], [-0.2, 0.4]])
        layer1_biases = np.array([[0.05, -0.05]])
        layer2_weights = np.array([[0.5], [-0.6]])
        layer2_biases = np.array([[0.1]])

        self.layer_configs = [
            {'input_size': 2, 'output_size': 2, 'activation_cls': Tanh, 'name': 'GizliTest',
                'weights': layer1_weights, 'biases': layer1_biases},
            {'input_size': 2, 'output_size': 1, 'activation_cls': Sigmoid, 'name': 'CiktiTest',
                'weights': layer2_weights, 'biases': layer2_biases}
        ]

        # Elle hesaplanan beklenen çıktı (setUp'taki ağırlıklarla)
        # H1_z = [[0.14, 0.02]]
        # H1_out = Tanh(H1_z) approx [[0.1390, 0.0199]]
        # Out_z = [[0.15756]]
        # Out_final = Sigmoid(Out_z) approx 0.53929
        self.expected_network_output_value = np.array([[0.53929]])

        # Test için y_true
        self.y_true_sample_for_loss = np.array([[0.8]]) # Sabit bir y_true değeri

    def test_complete_simple_network_flow(self):
        """
        Basit, elle tanımlanmış bir ağ için ileri yayılım, kayıp ve temel geri yayılımı
        tek bir test metodu içinde test eder.
        """
        logger.info("========== TEST: BASİT NETWORK AKIŞI (İLERİ/KAYIP/GERİ) ==========")

        # --- Ağ Oluşturma ---
        nn_test = Network()
        logger.info("Test Ağı Oluşturuluyor...")
        for i, config in enumerate(self.layer_configs):
            activation_instance = config['activation_cls']() if config['activation_cls'] else Linear()
            layer = DenseLayer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation_function=activation_instance,
                name=config.get('name', f"Layer{i+1}")
            )
            if 'weights' in config and config['weights'] is not None:
                self.assertEqual(layer.weights.shape, config['weights'].shape,
                                    f"{layer.name} için beklenen ağırlık şekli {layer.weights.shape}, verilen {config['weights'].shape}")
                layer.weights = config['weights'].copy()
            if 'biases' in config and config['biases'] is not None:
                self.assertEqual(layer.biases.shape, config['biases'].shape,
                                    f"{layer.name} için beklenen bias şekli {layer.biases.shape}, verilen {config['biases'].shape}")
                layer.biases = config['biases'].copy()
            nn_test.add_layer(layer)
            logger.info(f"  Eklendi: {layer} - Ağırlıklar {('elle ayarlandı' if 'weights' in config else 'rastgele')}")
        logger.info(f"Oluşturulan Ağ Mimarisi:\n{nn_test}")

        # --- İleri Yayılım Testi ---
        logger.info("--- İleri Yayılım (Forward Pass) ---")
        logger.info(f"Girdi Verisi:\n{self.input_data_sample}")
        network_output = nn_test.forward(self.input_data_sample.copy())
        logger.info(f"Ağ Çıktısı (shape {network_output.shape}):\n{network_output}")
        self.assertEqual(network_output.shape, self.expected_final_output_shape,
                            f"Hata: Ağ çıktısı boyutu yanlış! Beklenen: {self.expected_final_output_shape}, Gelen: {network_output.shape}")
        np.testing.assert_array_almost_equal(network_output, self.expected_network_output_value, decimal=4,
                                                err_msg="İleri yayılım sonucu beklenen değerden farklı.")
        logger.info("İleri yayılım başarılı (şekil ve değer).")

        # --- Kayıp Hesaplama Testi ---
        logger.info("--- Kayıp Hesaplama ---")
        loss_instance_test = self.loss_cls_for_test()
        logger.info(f"Kullanılan Kayıp Fonksiyonu: {loss_instance_test}")
        logger.info(f"Gerçek Değer (y_true_sample):\n{self.y_true_sample_for_loss}")

        loss_value = loss_instance_test.calculate(self.y_true_sample_for_loss, network_output)
        logger.info(f"Hesaplanan Kayıp Değeri: {loss_value:.6f}")
        self.assertIsInstance(loss_value, (float, np.floating), "Kayıp skaler olmalı.")

        loss_gradient = loss_instance_test.backward(self.y_true_sample_for_loss, network_output)
        logger.info(f"Kayıp Gradyanı (shape {loss_gradient.shape}):\n{loss_gradient}")
        self.assertEqual(loss_gradient.shape, network_output.shape,
                            "Kayıp gradyanı boyutu ağ çıktısı ile aynı olmalı.")
        logger.info("Kayıp hesaplama ve gradyanı başarılı.")

        # --- Geri Yayılım Testi (Basit Kontrol) ---
        logger.info("--- Geri Yayılım (Backward Pass) ---")
        weights_before_L0 = nn_test.layers[0].weights.copy()
        biases_before_L0 = nn_test.layers[0].biases.copy()
        weights_before_L1 = nn_test.layers[1].weights.copy()
        biases_before_L1 = nn_test.layers[1].biases.copy()

        nn_test.backward(loss_gradient, self.learning_rate_test)
        logger.info(f"Geri yayılım {self.learning_rate_test} öğrenme oranı ile yapıldı.")

        logger.info(f"{nn_test.layers[0].name} Ağırlıkları (güncelleme öncesi ilk): {weights_before_L0.flat[0]:.6f} -> (sonrası ilk): {nn_test.layers[0].weights.flat[0]:.6f}")
        self.assertFalse(np.allclose(weights_before_L0, nn_test.layers[0].weights),
                            f"{nn_test.layers[0].name} ağırlıkları değişmedi!")
        self.assertFalse(np.allclose(biases_before_L0, nn_test.layers[0].biases),
                            f"{nn_test.layers[0].name} biasları değişmedi!")

        logger.info(f"{nn_test.layers[1].name} Ağırlıkları (güncelleme öncesi ilk): {weights_before_L1.flat[0]:.6f} -> (sonrası ilk): {nn_test.layers[1].weights.flat[0]:.6f}")
        self.assertFalse(np.allclose(weights_before_L1, nn_test.layers[1].weights),
                            f"{nn_test.layers[1].name} ağırlıkları değişmedi!")
        self.assertFalse(np.allclose(biases_before_L1, nn_test.layers[1].biases),
                            f"{nn_test.layers[1].name} biasları değişmedi!")
        logger.info("Geri yayılım temel ağırlık güncelleme kontrolü tamamlandı.")
        logger.info("========== BASİT NETWORK AKIŞ TESTİ TAMAMLANDI ==========")

if __name__ == '__main__':
    unittest.main()