import unittest
import numpy as np
from DenseLayer import DenseLayer
from Activation import Sigmoid, Linear

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

if __name__ == '__main__':
    unittest.main()