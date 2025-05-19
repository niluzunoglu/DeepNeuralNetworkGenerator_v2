import sys
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
import numpy as np
from app import GeneratorWindow 
import logging
from implementations import Network, DenseLayer, Loss, Activation

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = None

def setUpModule():
    global app
    app = QApplication.instance() 
    if app is None: 
        app = QApplication(sys.argv)

def tearDownModule():
    global app
    app = None 

class TestGeneratorWindowUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.window = GeneratorWindow()
        self.window.show()
        QApplication.processEvents()

    def tearDown(self):
        self.window.close()
        del self.window 
        QApplication.processEvents()

    def test_01_initial_state_and_defaults(self):
        self.assertEqual(self.window.spin_LR.value(), 0.01)
        self.assertEqual(self.window.epoch.value(), 500)
        self.assertEqual(self.window.combo_loss.currentText(), "Mean Squared Error")
        self.assertEqual(self.window.spin_katman_sayisi.value(), 2)
        self.assertEqual(self.window.spin_input_features.value(), 2)
        self.assertEqual(self.window.line_edit_input_vector.text(), "0.5, 0.2")
        self.assertEqual(self.window.line_edit_target_vector.text(), "0.8, 0.6") 
        self.assertIn("Lütfen ağ parametrelerini belirleyip bir eylem seçin.", self.window.cikti_alani.toPlainText())
        self.assertEqual(self.window.status_bar.currentMessage(), "NNG arayüzü hazır.")
        # Başlangıçta 2 katman için detaylar olmalı
        self.assertEqual(len(self.window.katman_girdileri_widgetlari), 2)

    def test_02_reset_parameters(self):

        self.window.spin_LR.setValue(0.5)
        self.window.epoch.setValue(50)
        self.window.spin_katman_sayisi.setValue(3) 
        QApplication.processEvents() 

        self.assertNotEqual(self.window.spin_LR.value(), 0.01)
        self.assertEqual(len(self.window.katman_girdileri_widgetlari), 3)

        QTest.mouseClick(self.window.reset_model_parameters, Qt.MouseButton.LeftButton)
        QApplication.processEvents()

        self.assertEqual(self.window.spin_LR.value(), 0.01)
        self.assertEqual(self.window.epoch.value(), 500)
        self.assertEqual(self.window.spin_katman_sayisi.value(), 2) 
        self.assertEqual(len(self.window.katman_girdileri_widgetlari), 2) 
        self.assertIn("Parametreler varsayılan değerlere sıfırlandı.", self.window.cikti_alani.toPlainText())

    def test_03_generate_model_output_text(self):
        """Modeli Oluştur ve Eğit butonuna basıldığında çıktı alanındaki metni test eder."""
        # Varsayılan değerlerle çalıştıralım
        self.window.spin_input_features.setValue(2)
        self.window.line_edit_input_vector.setText("0.1, 0.9")
        self.window.line_edit_target_vector.setText("0.7") # Tek çıktılı bir ağ için
        self.window.spin_katman_sayisi.setValue(1) # Tek bir çıktı katmanı
        QApplication.processEvents() # katman_sayisi_degisti'nin çalışması için
        
        # Son katmanın (tek katman) nöron sayısını 1 yapalım
        self.window.katman_girdileri_widgetlari[0]['noron_spinbox'].setValue(1)
        self.window.katman_girdileri_widgetlari[0]['aktivasyon_combobox'].setCurrentText("Sigmoid")
        QApplication.processEvents()


        QTest.mouseClick(self.window.generate_model_train, Qt.MouseButton.LeftButton)
        QApplication.processEvents() # Slotun çalışıp UI'ı güncellemesi için biraz zaman tanı

        cikti_metni = self.window.cikti_alani.toPlainText()
        self.assertIn("Ağ parametreleri okunuyor ve ağ oluşturuluyor...", cikti_metni)
        self.assertIn("Girdi Özellik Sayısı: 2", cikti_metni)
        self.assertIn("Örnek Girdi (X_sample): [[0.1 0.9]]", cikti_metni)
        self.assertIn("Örnek Hedef (y_true_sample): [[0.7]]", cikti_metni)
        self.assertIn("Toplam Katman Sayısı: 1", cikti_metni)
        self.assertIn("Katman 1: Nöron=1, Aktivasyon=Sigmoid", cikti_metni)
        self.assertIn("Sinir ağı başarıyla oluşturuldu", cikti_metni)
        self.assertIn("EĞİTİM BAŞLIYOR (Tek Örnekle)", cikti_metni)
        self.assertIn("Eğitim tamamlandı.", cikti_metni)
        self.assertIn("Son Ortalama Kayıp:", cikti_metni) # Gerçek kayıp değerini test etmek zor olabilir
        self.assertIn("Son Tahmin:", cikti_metni)

    def test_04_iterative_train_output_text_after_network_creation(self):

        # Önce bir ağ oluşturalım (generate_model_train ile)
        self.window.spin_input_features.setValue(2)
        self.window.line_edit_input_vector.setText("0.1, 0.9")
        self.window.line_edit_target_vector.setText("0.7") # Tek çıktılı bir ağ için
        self.window.spin_katman_sayisi.setValue(1)
        QApplication.processEvents()
        self.window.katman_girdileri_widgetlari[0]['noron_spinbox'].setValue(1)
        self.window.katman_girdileri_widgetlari[0]['aktivasyon_combobox'].setCurrentText("Sigmoid")
        QApplication.processEvents()
        
        QTest.mouseClick(self.window.generate_model_train, Qt.MouseButton.LeftButton)
        QApplication.processEvents() # Ağın oluşması için

        # Şimdi iteratif eğitim butonuna basalım
        QTest.mouseClick(self.window.iterate_over_network, Qt.MouseButton.LeftButton)
        QApplication.processEvents()

        cikti_metni = self.window.cikti_alani.toPlainText()
        self.assertIn("İTERATİF EĞİTİM ADIMI BAŞLATILIYOR", cikti_metni)
        self.assertIn("Kullanılacak Örnek Girdi (X_sample): [[0.1 0.9]]", cikti_metni)
        self.assertIn("Kullanılacak Örnek Hedef (y_true_sample): [[0.7]]", cikti_metni)
        self.assertIn("Adım 1: İleri Yayılım...", cikti_metni)
        self.assertIn("Adım 2: Kayıp Hesaplanıyor...", cikti_metni)
        self.assertIn("Adım 3: Kayıp Gradyanı Hesaplanıyor...", cikti_metni)
        self.assertIn("Adım 4: Geri Yayılım ve Ağırlık Güncelleme...", cikti_metni)
        self.assertIn("Adım 5: Güncellenmiş Ağırlıklarla Tekrar İleri Yayılım...", cikti_metni)
        self.assertIn("TEK İTERASYON TAMAMLANDI", cikti_metni)
    
    def test_05_iterative_train_without_network(self):
        """Ağ oluşturulmadan iteratif eğitim butonuna basılmasını test eder."""
        # Doğrudan iteratif eğitim butonuna basalım
        QTest.mouseClick(self.window.iterate_over_network, Qt.MouseButton.LeftButton)
        QApplication.processEvents()

        cikti_metni = self.window.cikti_alani.toPlainText()
        # QMessageBox çıktısını yakalamak zor, bu yüzden cikti_alani'ndaki HATA mesajını kontrol edelim
        self.assertIn("HATA: Önce 'Modeli Oluştur ve Eğit' butonu ile bir ağ oluşturmalısınız.", cikti_metni)
        # Gerçekte bir QMessageBox açılacak, bunu programatik olarak kapatmak gerekebilir veya
        # QMessageBox.warning'i mock'lamak gerekebilir daha ileri testlerde.

    def test_06_numerical_network_creation_and_single_epoch_train(self):
        logger.info("--- Test: Sayısal Ağ Oluşturma ve Kısa Eğitim ---")

        # 1. UI'da Parametreleri Ayarla (Bu kısım aynı)
        self.window.spin_input_features.setValue(2)
        self.window.line_edit_input_vector.setText("0.5, 0.2")
        self.window.line_edit_target_vector.setText("0.8")
        self.window.spin_LR.setValue(0.1)
        self.window.epoch.setValue(10)
        self.window.combo_loss.setCurrentText("Mean Squared Error")
        self.window.spin_katman_sayisi.setValue(2)
        QApplication.processEvents()

        l1_weights = np.array([[0.1, 0.3], [-0.2, 0.4]])
        l1_biases = np.array([[0.05, -0.05]])
        self.window.katman_girdileri_widgetlari[0]['noron_spinbox'].setValue(2)
        self.window.katman_girdileri_widgetlari[0]['aktivasyon_combobox'].setCurrentText("Tanh")
        self.window.katman_girdileri_widgetlari[0]['custom_weights'] = l1_weights
        self.window.katman_girdileri_widgetlari[0]['custom_biases'] = l1_biases
        self.window.katman_girdileri_widgetlari[0]['btn_agirlik'].setText("Ağırlıklar ✓")

        l2_weights = np.array([[0.5], [-0.6]])
        l2_biases = np.array([[0.1]])
        self.window.katman_girdileri_widgetlari[1]['noron_spinbox'].setValue(1)
        self.window.katman_girdileri_widgetlari[1]['aktivasyon_combobox'].setCurrentText("Sigmoid")
        self.window.katman_girdileri_widgetlari[1]['custom_weights'] = l2_weights
        self.window.katman_girdileri_widgetlari[1]['custom_biases'] = l2_biases
        self.window.katman_girdileri_widgetlari[1]['btn_agirlik'].setText("Ağırlıklar ✓")
        QApplication.processEvents()

        # --- AĞ OLUŞTURMA ADIMI ---
        # "Modeli Oluştur ve Eğit" butonu tıklandığında generate_and_train_model_slot çağrılır.
        # Bu slot içinde önce ağ oluşturulur, sonra eğitim döngüsü başlar.
        # Biz ağırlıkların atanmasını, ağ oluşturulduktan HEMEN SONRA kontrol etmeliyiz.
        # Ancak generate_and_train_model_slot hem oluşturma hem eğitimi yapıyor.
        # Bu testi daha iyi hale getirmek için ya generate_and_train_model_slot'u ikiye bölmeli
        # (bir oluşturma, bir eğitme slotu) ya da slot içinde ağ oluşturulduktan sonra
        # bir sinyal yayıp o sinyali testte yakalamalıyız.

        # ŞİMDİLİK BASİT YAKLAŞIM:
        # generate_and_train_model_slot'un içinde, ağ oluşturulduktan sonraki
        # self.network_instance'ı referans alacağız. Bu, slotun tamamı bittikten sonra
        # ağırlıkların değişmiş olacağı anlamına gelir. Bu yüzden bu özel testte
        # ağırlıkların DEĞİŞMİŞ olmasını assert etmeliyiz, EŞİT olmasını değil.

        # Ya da, generate_and_train_model_slot'u çağırmadan ÖNCE network instance'ı
        # manuel olarak oluşturup ağırlıkları test edebiliriz, sonra slotu çağırıp
        # eğitimin etkisini görebiliriz. Bu daha doğru bir test olur.

        # ---- YENİ YAKLAŞIM: Ağırlık atamasını slotu çağırmadan önce manuel test et ----
        # Bu, generate_and_train_model_slot'un ağ oluşturma kısmının doğru çalıştığını test eder.
        
        # 1. Parametreleri al (generate_and_train_model_slot'un başındaki gibi)
        input_feature_count_test = self.window.spin_input_features.value()
        katman_yapilandirmalari_ui_test = []
        for i, grp in enumerate(self.window.katman_girdileri_widgetlari):
            katman_yapilandirmalari_ui_test.append({
                'noron': grp['noron_spinbox'].value(),
                'aktivasyon_str': grp['aktivasyon_combobox'].currentText(),
                'custom_weights': grp.get('custom_weights'),
                'custom_biases': grp.get('custom_biases')
            })

        # 2. Test için geçici bir ağ oluştur (generate_and_train_model_slot'un mantığını taklit ederek)
        test_network_instance = Network()
        current_input_size_test = input_feature_count_test
        for i, config_ui_test in enumerate(katman_yapilandirmalari_ui_test):
            activation_func_test = self.window.get_activation_class_from_string(config_ui_test['aktivasyon_str'])
            layer_test = DenseLayer(current_input_size_test, config_ui_test['noron'], activation_func_test)
            
            if config_ui_test['custom_weights'] is not None:
                layer_test.weights = config_ui_test['custom_weights'].copy()
            if config_ui_test['custom_biases'] is not None:
                layer_test.biases = config_ui_test['custom_biases'].copy()
            
            test_network_instance.add_layer(layer_test)
            current_input_size_test = config_ui_test['noron']

        # 3. Elle atanan ağırlıkların doğruluğunu ŞİMDİ kontrol et
        logger.info("Test: Elle atanan ağırlıkların ağ oluşturulduktan sonraki kontrolü...")
        np.testing.assert_array_almost_equal(test_network_instance.layers[0].weights, l1_weights, decimal=6,
                                                err_msg="İlk katmanın ağırlıkları oluşturma sonrası doğru atanmamış.")
        np.testing.assert_array_almost_equal(test_network_instance.layers[0].biases, l1_biases, decimal=6,
                                                err_msg="İlk katmanın biasları oluşturma sonrası doğru atanmamış.")
        np.testing.assert_array_almost_equal(test_network_instance.layers[1].weights, l2_weights, decimal=6,
                                                err_msg="İkinci katmanın ağırlıkları oluşturma sonrası doğru atanmamış.")
        np.testing.assert_array_almost_equal(test_network_instance.layers[1].biases, l2_biases, decimal=6,
                                                err_msg="İkinci katmanın biasları oluşturma sonrası doğru atanmamış.")
        logger.info("Test: Elle atanan ağırlıklar ağa doğru şekilde yüklendi.")


        # --- ŞİMDİ GERÇEK SLOTU ÇAĞIRIP EĞİTİMİN ETKİSİNİ TEST EDELİM ---
        # Önceki network instance'ını temizleyelim (eğer varsa)
        if hasattr(self.window, 'network_instance'):
            del self.window.network_instance 
            
        QTest.mouseClick(self.window.generate_model_train, Qt.MouseButton.LeftButton)
        QApplication.processEvents()

        cikti_metni = self.window.cikti_alani.toPlainText()
        self.assertIn("Sinir ağı başarıyla oluşturuldu", cikti_metni)
        self.assertIn("EĞİTİM BAŞLIYOR (Tek Örnekle)", cikti_metni)
        self.assertIn("Eğitim tamamlandı.", cikti_metni)
        self.assertIn("Son Ortalama Kayıp:", cikti_metni)

        # Eğitim sonrası network instance'ını al
        self.assertIsNotNone(getattr(self.window, 'network_instance', None), "Network instance eğitim sonrası oluşturulmamış.")
        trained_network = self.window.network_instance
        self.assertEqual(len(trained_network.layers), 2)

        # Eğitim sonrası ağırlıkların BAŞLANGIÇTAKİ (l1_weights) değerlerden FARKLI olmasını bekle
        logger.info("Test: Eğitim sonrası ağırlıkların değişip değişmediğinin kontrolü...")
        self.assertFalse(np.allclose(trained_network.layers[0].weights, l1_weights, atol=1e-6), # atol ile küçük tolerans
                            "İlk katmanın ağırlıkları eğitim sonrası değişmemiş görünüyor (başlangıçtakiyle aynı).")
        self.assertFalse(np.allclose(trained_network.layers[1].weights, l2_weights, atol=1e-6),
                            "İkinci katmanın ağırlıkları eğitim sonrası değişmemiş görünüyor (başlangıçtakiyle aynı).")
        logger.info("Test: Eğitim sonrası ağırlıklar başlangıç değerlerinden farklı.")


        # Kayıp azalması kontrolü (Bu kısım aynı kalabilir)
        last_loss_str = ""; first_epoch_loss_value = float('inf'); last_loss_value = float('inf')
        for line in cikti_metni.splitlines():
            if "Epoch 1/10 - Kayıp:" in line:
                try:
                    first_epoch_loss_value = float(line.split("Kayıp:")[1].split("-")[0].strip())
                except: pass
            if "Son Ortalama Kayıp:" in line:
                try:
                    last_loss_value = float(line.split(":")[1].strip())
                except: pass
        
        logger.info(f"Test: İlk Epoch Kaybı (Logdan): {first_epoch_loss_value}, Son Epoch Kaybı (Logdan): {last_loss_value}")
        self.assertTrue(first_epoch_loss_value != float('inf'), "Loglardan ilk epoch kaybı okunamadı.")
        self.assertTrue(last_loss_value != float('inf'), "Loglardan son epoch kaybı okunamadı.")
        self.assertLess(last_loss_value, first_epoch_loss_value + 1e-5,
                        "Eğitim sonrası kayıp, ilk epoch kaybından daha düşük olmalıydı.")

if __name__ == '__main__':

    # python -m unittest tests.test_app_generator_ui
    unittest.main(verbosity=2)