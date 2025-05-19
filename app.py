import sys
from PyQt6.QtWidgets import (
    QApplication,QMainWindow,
    QWidget,QVBoxLayout,
    QLabel,
    QStatusBar,QMenuBar,
    QPushButton, QLineEdit,    
    QTextEdit,    
    QComboBox,     
    QSpinBox,  
    QGroupBox,     
    QFormLayout,    
    QHBoxLayout     
)
from PyQt6.QtGui import QAction 
                               
from PyQt6.QtCore import Qt    
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QScrollArea

import logging
from stylesheets import dark_stylesheet

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QStyle, QToolButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize
from functools import partial
from helper_text import yardim_metni

from implementations import DenseLayer, Network, Activation, Loss
from implementations.Activation import ReLU, Sigmoid, Tanh, Linear
from implementations.Loss import MeanSquaredError, MeanAbsoluteError

from WeightBiasDialog import WeightBiasDialog

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class GeneratorWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Generator")
        self.setGeometry(100, 100, 850, 800) 
        self.katman_girdileri_widgetlari = []
        self._create_ui()
        self.varsayilan_degerleri_ayarla() 
        
    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        self._clear_layout(sub_layout)

    def katman_sayisi_degisti(self, yeni_katman_sayisi):

        logger.info(f"Katman sayısı değişti: {yeni_katman_sayisi}")

        self._clear_layout(self.layout_katman_detaylari)
        self.katman_girdileri_widgetlari = [] 

        for i in range(yeni_katman_sayisi):
            katman_numarasi = i + 1
            katman_layout = QHBoxLayout()

            label_noron = QLabel(f"Katman {katman_numarasi} Nöron Sayısı:")
            spin_noron_sayisi = QSpinBox()
            spin_noron_sayisi.setMinimum(1)
            spin_noron_sayisi.setMaximum(1024) 
            spin_noron_sayisi.setValue(10) 
            spin_noron_sayisi.setToolTip(f"Katman {katman_numarasi} için nöron sayısı.")
            
            katman_layout.addWidget(label_noron)
            katman_layout.addWidget(spin_noron_sayisi)
            katman_layout.addSpacing(20) 

            label_aktivasyon = QLabel("Aktivasyon:")
            combo_aktivasyon = QComboBox()
            combo_aktivasyon.addItems(["Sigmoid", "Tanh", "ReLU", "Linear"]) 
            if katman_numarasi == yeni_katman_sayisi and yeni_katman_sayisi > 1 : 
                 combo_aktivasyon.setCurrentText("Sigmoid") 
            elif yeni_katman_sayisi > 1: 
                 combo_aktivasyon.setCurrentText("ReLU")
            else:
                combo_aktivasyon.setCurrentText("Sigmoid")

            combo_aktivasyon.setToolTip(f"Katman {katman_numarasi} için aktivasyon fonksiyonu.")

            katman_layout.addWidget(label_aktivasyon)
            katman_layout.addWidget(combo_aktivasyon)
            katman_layout.addStretch()

            btn_agirlik_ayarla = QPushButton("Ağırlık ve Bias Ayarla")
            btn_agirlik_ayarla.setToolTip(f"Katman {katman_numarasi} için ağırlık ve bias değerlerini elle girin.")
            btn_agirlik_ayarla.clicked.connect(partial(self.agirlik_bias_ayarla_slot, katman_numarasi - 1)) # index olarak gönderelim
            katman_layout.addWidget(btn_agirlik_ayarla)

            katman_layout.addStretch()
            self.layout_katman_detaylari.addLayout(katman_layout)
            self.katman_girdileri_widgetlari.append({
                'noron_spinbox': spin_noron_sayisi,
                'aktivasyon_combobox': combo_aktivasyon,
                'btn_agirlik': btn_agirlik_ayarla,
                'custom_weights': None, 
                'custom_biases': None  
            })

    def parametreleri_sifirla_slot(self):

        logger.info("'Parametreleri Sıfırla' butonuna tıklandı.")
        self.varsayilan_degerleri_ayarla()

        # TODO: Katman detayları da varsayılan katman sayısına göre güncellenmeli
        self.katman_sayisi_degisti(self.spin_katman_sayisi.value())

        self.cikti_alani.setText("Parametreler varsayılan değerlere sıfırlandı.\nYeni bir ağ tanımlayabilirsiniz.")
        self.status_bar.showMessage("Parametreler sıfırlandı.")

    def varsayilan_degerleri_ayarla(self):

        if hasattr(self, 'line_edit_target_vector'): self.line_edit_target_vector.setText("0.8, 0.6")
        if hasattr(self, 'line_edit_input_vector'): self.line_edit_input_vector.setText("0.5, 0.2")
        if hasattr(self, 'spin_input_features'): self.spin_input_features.setValue(2) 
        if hasattr(self, 'spin_LR'): self.spin_LR.setValue(0.01)
        if hasattr(self, 'epoch'): self.epoch.setValue(1000)
        if hasattr(self, 'combo_loss'): self.combo_loss.setCurrentIndex(0) 
        if hasattr(self, 'spin_katman_sayisi'): self.spin_katman_sayisi.setValue(2) 
        if hasattr(self, 'combo_ogrenme_sekli'): self.combo_ogrenme_sekli.setCurrentIndex(0)
        if hasattr(self, 'spin_batch_boyutu'): self.spin_batch_boyutu.setValue(32)
        if hasattr(self, 'combo_ogrenme_sekli'): self.ogrenme_sekli_degisti_slot(self.combo_ogrenme_sekli.currentIndex())
        logger.info("Giriş alanları varsayılan değerlere ayarlandı.")

    def parse_input_vector(self, vector_str, expected_features):
        """Verilen string'i NumPy array'ine dönüştürür ve boyutunu kontrol eder."""
        if not vector_str.strip():
            return None, "Girdi vektörü boş bırakılamaz."
        try:
            # Birden fazla virgül veya boşlukla ayrılma durumlarını da handle edebiliriz
            # Örneğin: items = [float(x.strip()) for x in vector_str.replace(" ", "").split(',') if x.strip()]
            items = [float(x.strip()) for x in vector_str.split(',') if x.strip()]
            if not items: # Hepsi boşluk veya geçersizse
                 return None, "Geçerli sayısal değer bulunamadı."

            input_array = np.array([items]) # Shape (1, num_features) olması için çift köşeli parantez
            if input_array.shape[1] != expected_features:
                return None, f"Girdi vektörü {expected_features} özellik içermeli, {input_array.shape[1]} özellik girildi."
            return input_array, None # Başarılı: array ve None (hata yok)
        except ValueError:
            return None, "Girdi vektörü sadece virgülle ayrılmış sayılar içermelidir."
        except Exception as e:
            return None, f"Girdi vektörü işlenirken hata: {e}"


    def get_activation_class_from_string(self, activation_str_ui):
        """Verilen string'e karşılık gelen aktivasyon sınıfının bir ÖRNEĞİNİ döndürür."""
        # Bu fonksiyon, implementasyonlarınızdaki gerçek aktivasyon sınıflarının
        # bir örneğini (instance) döndürmeli.
        # Örn: implementations.activations.ReLU(), implementations.activations.Sigmoid()
        if activation_str_ui == "ReLU": return ReLU()
        elif activation_str_ui == "Sigmoid": return Sigmoid()
        elif activation_str_ui == "Tanh": return Tanh()
        elif activation_str_ui == "Linear": return Linear()
        else:
            logger.warning(f"Bilinmeyen aktivasyon fonksiyonu: {activation_str_ui}. Linear kullanılacak.")
            return Linear()

    def get_loss_class_from_string(self, loss_str_ui):
        """Verilen string'e karşılık gelen kayıp fonksiyonu sınıfının bir ÖRNEĞİNİ döndürür."""
        # Bu fonksiyon, implementasyonlarınızdaki gerçek kayıp sınıflarının
        # bir örneğini (instance) döndürmeli.
        # Örn: implementations.losses.MeanSquaredError()
        if loss_str_ui == "Mean Squared Error": return MeanSquaredError()
        elif loss_str_ui == "Mean Absolute Error": return MeanAbsoluteError() # İsmin tam eşleştiğinden emin ol
        else:
            logger.warning(f"Bilinmeyen kayıp fonksiyonu: {loss_str_ui}. MeanSquaredError kullanılacak.")
            return MeanSquaredError()

    def generate_and_train_model_slot(self):

        self.cikti_alani.clear()
        self.cikti_alani.append("Ağ parametreleri okunuyor ve ağ oluşturuluyor...\n" + "="*50 + "\n")

        
        ogrenme_orani = self.spin_LR.value()
        epoch_sayisi = self.epoch.value()
        secilen_loss_fonksiyonu_str = self.combo_loss.currentText()
        toplam_katman_sayisi = self.spin_katman_sayisi.value()
        secilen_ogrenme_sekli_str = self.combo_ogrenme_sekli.currentText()

        batch_boyutu_degeri = self.spin_batch_boyutu.value() if self.spin_batch_boyutu.isVisible() else 1
        input_feature_count = self.spin_input_features.value()
        input_vector_str = self.line_edit_input_vector.text()
        target_vector_str = self.line_edit_target_vector.text()

        X_sample, error_msg_x = self.parse_input_vector(input_vector_str, input_feature_count)
        if error_msg_x:
            self.cikti_alani.append(f"ÖRNEK GİRDİ HATASI: {error_msg_x}")
            QMessageBox.warning(self, "Girdi Hatası", f"Örnek Girdi: {error_msg_x}")
            return

        # Hedef çıktı vektörünün boyutunu belirlemek için geçici bir ağ oluşturup son katman nöron sayısını alabiliriz
        # VEYA kullanıcıdan çıktı nöron sayısını da alabiliriz.
        # Şimdilik, katman yapılandırmasından son katmanın nöron sayısını almayı deneyelim.
        # Bu, katman_girdileri_widgetlari'nın doğru sayıda elemana sahip olmasını gerektirir.
        if len(self.katman_girdileri_widgetlari) != toplam_katman_sayisi or not self.katman_girdileri_widgetlari:
            msg = "HATA: Katman detayları eksik veya yanlış. 'Toplam Katman Sayısı'nı kontrol edin."
            self.cikti_alani.append(msg); QMessageBox.critical(self, "Yapılandırma Hatası", msg); return
        
        try:
            output_layer_neurons_ui = self.katman_girdileri_widgetlari[-1]['noron_spinbox'].value()
        except (IndexError, KeyError):
            self.cikti_alani.append("HATA: Çıktı katmanı nöron sayısı belirlenemedi.")
            return

        y_true_sample, error_msg_y = self.parse_input_vector(target_vector_str, output_layer_neurons_ui)
        if error_msg_y:
            self.cikti_alani.append(f"ÖRNEK HEDEF ÇIKTI HATASI: {error_msg_y}")
            QMessageBox.warning(self, "Girdi Hatası", f"Örnek Hedef Çıktı: {error_msg_y}")
            return

        self.cikti_alani.append("Genel Parametreler:")
        self.cikti_alani.append(f"  Girdi Özellik Sayısı: {input_feature_count}")
        self.cikti_alani.append(f"  Örnek Girdi (X_sample): {X_sample}")
        self.cikti_alani.append(f"  Örnek Hedef (y_true_sample): {y_true_sample}")
        self.cikti_alani.append(f"  Öğrenme Oranı: {ogrenme_orani}")
        # ... (diğer genel parametreleri yazdırma) ...
        self.cikti_alani.append(f"  Epoch Sayısı: {epoch_sayisi}"); self.cikti_alani.append(f"  Kayıp Fonksiyonu: {secilen_loss_fonksiyonu_str}"); self.cikti_alani.append(f"  Toplam Katman Sayısı: {toplam_katman_sayisi}"); self.cikti_alani.append(f"  Öğrenme Şekli: {secilen_ogrenme_sekli_str}");
        if batch_boyutu_degeri is not None and "Mini-batch" in secilen_ogrenme_sekli_str: self.cikti_alani.append(f"  Batch Boyutu: {batch_boyutu_degeri}")
        self.cikti_alani.append("\n" + "="*50 + "\n")

        # 2. Katman Detaylarını Oku (Bu kısım aynı kalabilir)
        # ... (katman_yapilandirmalari_ui oluşturma kısmı) ...
        self.cikti_alani.append("Katman Detayları:"); katman_yapilandirmalari_ui = []
        for i, grp in enumerate(self.katman_girdileri_widgetlari):
            kat_no = i + 1; n = grp['noron_spinbox'].value(); a_str = grp['aktivasyon_combobox'].currentText(); cw = grp.get('custom_weights'); cb = grp.get('custom_biases')
            self.cikti_alani.append(f"  Katman {kat_no}: Nöron={n}, Aktivasyon={a_str}{' (Özel W)' if cw is not None else ''}{' (Özel B)' if cb is not None else ''}")
            katman_yapilandirmalari_ui.append({'noron': n, 'aktivasyon_str': a_str, 'custom_weights': cw, 'custom_biases': cb})
        self.cikti_alani.append("="*50 + "\nParametreler başarıyla okundu.")
        self.cikti_alani.append("Sinir ağı oluşturuluyor...")


        # === AĞ OLUŞTURMA (GERÇEK IMPLEMENTASYON) ===
        try:
            self.network_instance = Network()
            current_input_size = input_feature_count

            for i, config_ui in enumerate(katman_yapilandirmalari_ui):
                activation_function_instance = self.get_activation_class_from_string(config_ui['aktivasyon_str'])
                new_dense_layer = DenseLayer(current_input_size, config_ui['noron'], activation_function=activation_function_instance, name=f"Katman_{i+1}")
                if config_ui['custom_weights'] is not None and new_dense_layer.weights.shape == config_ui['custom_weights'].shape:
                    new_dense_layer.weights = config_ui['custom_weights'].copy()
                if config_ui['custom_biases'] is not None and new_dense_layer.biases.shape == config_ui['custom_biases'].shape:
                    new_dense_layer.biases = config_ui['custom_biases'].copy()
                self.network_instance.add_layer(new_dense_layer)
                current_input_size = config_ui['noron']
            
            self.cikti_alani.append(f"\nSinir ağı başarıyla oluşturuldu:\n{self.network_instance}")
            self.status_bar.showMessage("Ağ oluşturuldu. Eğitim başlıyor...")

            # === EĞİTİM DÖNGÜSÜ (TEK ÖRNEKLE) ===
            self.cikti_alani.append("\n" + "="*10 + " EĞİTİM BAŞLIYOR (Tek Örnekle) " + "="*10)
            if "GD" not in secilen_ogrenme_sekli_str: # Sadece uyarı
                 self.cikti_alani.append(f"UYARI: Tek örnekle eğitim yapıldığı için '{secilen_ogrenme_sekli_str}' pratikte SGD gibi davranacaktır.")

            loss_function_instance = self.get_loss_class_from_string(secilen_loss_fonksiyonu_str)
            
            # Kayıp geçmişini tutmak için
            loss_history = []

            # Başlangıçtaki ağırlıkları göstermek için (isteğe bağlı)
            # self.cikti_alani.append("\nBaşlangıç Ağırlıkları (ilk katman, ilk ağırlık):")
            # self.cikti_alani.append(str(self.network_instance.layers[0].weights.flat[0]))

            for epoch in range(epoch_sayisi):
                # 1. İleri Yayılım
                y_pred = self.network_instance.forward(X_sample)

                # 2. Kayıp Hesaplama
                loss = loss_function_instance.calculate(y_true_sample, y_pred)
                loss_history.append(loss)

                # 3. Kayıp Gradyanı
                loss_grad = loss_function_instance.backward(y_true_sample, y_pred)

                # 4. Geri Yayılım
                self.network_instance.backward(loss_grad, ogrenme_orani)

                # Belirli aralıklarla log yazdır
                if (epoch + 1) % (max(1, epoch_sayisi // 10)) == 0 or epoch == 0 or epoch == epoch_sayisi -1 : # Yaklaşık 10 log + ilk ve son
                    self.cikti_alani.append(f"Epoch {epoch+1}/{epoch_sayisi} - Kayıp: {loss:.8f} - Tahmin: {y_pred[0] if y_pred.size==1 else y_pred}")
                    QApplication.processEvents() # Arayüzün güncellenmesi için (uzun eğitimlerde önemli)

            self.cikti_alani.append("\nEğitim tamamlandı.")
            self.cikti_alani.append(f"Son Ortalama Kayıp: {loss_history[-1]:.8f}")
            self.cikti_alani.append(f"Son Tahmin: {self.network_instance.predict(X_sample)}")

            # Son ağırlıkları göstermek için (isteğe bağlı)
            # self.cikti_alani.append("\nSon Ağırlıklar (ilk katman, ilk ağırlık):")
            # self.cikti_alani.append(str(self.network_instance.layers[0].weights.flat[0]))

            self.status_bar.showMessage("Eğitim tamamlandı.")

        except Exception as e:
            logger.error(f"Ağ oluşturma veya eğitim sırasında bir hata oluştu: {e}", exc_info=True)
            self.cikti_alani.append(f"\nHATA: Ağ oluşturma veya eğitim sırasında bir sorun oluştu.\nDetaylar için loglara bakın.\n{e}")
            self.status_bar.showMessage("Hata: Ağ oluşturma/eğitim başarısız.")

    def iteratif_egitim_slot(self):
        logger.info("'Modeli Eğit (İteratif)' butonuna tıklandı.")
        self.cikti_alani.clear()
        self.cikti_alani.append("\n" + "="*50 + "\nİTERATİF EĞİTİM ADIMI BAŞLATILIYOR\n" + "="*50)

        # 1. Ağın varlığını kontrol et
        if not hasattr(self, 'network_instance') or self.network_instance is None or not self.network_instance.layers:
            self.cikti_alani.append("HATA: Önce 'Modeli Oluştur ve Eğit' butonu ile bir ağ oluşturmalısınız.")
            QMessageBox.warning(self, "Ağ Eksik", "Lütfen önce bir ağ modeli oluşturun.")
            return

        # 2. Girdi ve Hedef Çıktı Vektörlerini Oku
        try:
            input_feature_count = self.spin_input_features.value()
            input_vector_str = self.line_edit_input_vector.text()
            target_vector_str = self.line_edit_target_vector.text() # <<< YENİ
        except AttributeError:
            self.cikti_alani.append("HATA: Girdi/Hedef için UI elemanları bulunamadı.")
            return

        X_sample, error_msg_x = self.parse_input_vector(input_vector_str, input_feature_count)
        if error_msg_x:
            self.cikti_alani.append(f"ÖRNEK GİRDİ HATASI: {error_msg_x}")
            QMessageBox.warning(self, "Girdi Hatası", f"Örnek Girdi: {error_msg_x}")
            return
        
        # Hedef çıktı vektörünün boyutunu ağın son katmanının nöron sayısına göre belirle
        try:
            output_layer_neurons = self.network_instance.layers[-1].output_size
        except IndexError:
            self.cikti_alani.append("HATA: Ağda katman bulunmuyor gibi görünüyor.")
            return
            
        y_true_sample, error_msg_y = self.parse_input_vector(target_vector_str, output_layer_neurons) # parse_input_vector'u yeniden kullan
        if error_msg_y:
            self.cikti_alani.append(f"ÖRNEK HEDEF ÇIKTI HATASI: {error_msg_y}")
            QMessageBox.warning(self, "Girdi Hatası", f"Örnek Hedef Çıktı: {error_msg_y}")
            return
        
        self.cikti_alani.append(f"Kullanılacak Örnek Girdi (X_sample): {X_sample}")
        self.cikti_alani.append(f"Kullanılacak Örnek Hedef (y_true_sample): {y_true_sample}")

        # Gerekli diğer parametreler
        ogrenme_orani = self.spin_LR.value()
        secilen_loss_fonksiyonu_str = self.combo_loss.currentText()
        loss_function_instance = self.get_loss_class_from_string(secilen_loss_fonksiyonu_str)

        self.cikti_alani.append("\n--- TEK İTERASYON BAŞLIYOR ---")
        try:
            # ... (İleri yayılım, kayıp hesaplama, gradyan, geri yayılım adımları aynı) ...
            self.cikti_alani.append("\nAdım 1: İleri Yayılım...")
            y_pred = self.network_instance.forward(X_sample)
            self.cikti_alani.append(f"  Ağ Tahmini (y_pred): {y_pred}")

            self.cikti_alani.append("\nAdım 2: Kayıp Hesaplanıyor...")
            loss = loss_function_instance.calculate(y_true_sample, y_pred)
            self.cikti_alani.append(f"  Hesaplanan Kayıp: {loss:.6f}")

            self.cikti_alani.append("\nAdım 3: Kayıp Gradyanı Hesaplanıyor...")
            loss_grad = loss_function_instance.backward(y_true_sample, y_pred)
            self.cikti_alani.append(f"  Kayıp Gradyanı (dL/dy_pred): {loss_grad}")

            self.cikti_alani.append("\nAdım 4: Geri Yayılım ve Ağırlık Güncelleme...")
            self.network_instance.backward(loss_grad, ogrenme_orani)
            self.cikti_alani.append("  Ağırlıklar ve biaslar güncellendi.")
            
            # Ağırlıkları yazdırmak isteğe bağlı (çok fazla çıktı üretebilir)
            # for i, layer in enumerate(self.network_instance.layers):
            #     self.cikti_alani.append(f"  Güncellenmiş Ağırlıklar - Katman {i+1} (ilk eleman): {layer.weights.flat[0]:.4f}")

            self.cikti_alani.append("\nAdım 5: Güncellenmiş Ağırlıklarla Tekrar İleri Yayılım...")
            y_pred_after = self.network_instance.forward(X_sample)
            loss_after = loss_function_instance.calculate(y_true_sample, y_pred_after)
            self.cikti_alani.append(f"  Yeni Tahmin: {y_pred_after}")
            self.cikti_alani.append(f"  Yeni Kayıp: {loss_after:.6f}")

            if np.isclose(loss_after, loss) or loss_after < loss : # Kayıp azalmalı veya çok yakın kalmalı
                self.cikti_alani.append("  BAŞARILI: Bir iterasyon sonrası kayıp azaldı veya değişmedi (çok küçük gradyan).")
            else:
                self.cikti_alani.append("  UYARI: Bir iterasyon sonrası kayıp arttı!")
            
            self.cikti_alani.append("\n--- TEK İTERASYON TAMAMLANDI ---")
            self.status_bar.showMessage("Tek iterasyon tamamlandı.")

        except Exception as e:
            logger.error(f"İteratif eğitim sırasında hata: {e}", exc_info=True)
            self.cikti_alani.append(f"\nHATA: İteratif eğitim sırasında bir sorun oluştu.\n{e}")
            self.status_bar.showMessage("Hata: İteratif eğitim başarısız.")

    def yardim_penceresi_goster_slot(self):

        logger.info("Yardım penceresi gösteriliyor.")
        yardim_basligi = "Neural Network Generator - Yardım"
        QMessageBox.information(self, yardim_basligi, yardim_metni)

    def ogrenme_sekli_degisti_slot(self, index):

        secilen_sekil = self.combo_ogrenme_sekli.currentText()
        logger.info(f"Öğrenme şekli değişti: {secilen_sekil}")

        if "Mini-batch" in secilen_sekil:
            self.label_batch_boyutu.setVisible(True)
            self.spin_batch_boyutu.setVisible(True)
        else:
            self.label_batch_boyutu.setVisible(False)
            self.spin_batch_boyutu.setVisible(False)

    def agirlik_bias_ayarla_slot(self, katman_index):

        logger.info(f"Katman {katman_index + 1} için ağırlık/bias ayarlama butonu tıklandı.")
                
        if katman_index == 0:
            if hasattr(self, 'spin_input_features'):
                prev_layer_neurons = self.spin_input_features.value()
            else:
                logger.error("Girdi Özellik Sayısı spinbox'ı bulunamadı!")
                QMessageBox.critical(self, "Hata", "Girdi Özellik Sayısı belirlenemedi.")
                return
        else:
            prev_layer_neurons = self.katman_girdileri_widgetlari[katman_index - 1]['noron_spinbox'].value()

        current_layer_neurons = self.katman_girdileri_widgetlari[katman_index]['noron_spinbox'].value()

        dialog = WeightBiasDialog(katman_index + 1, prev_layer_neurons, current_layer_neurons, self)
        
        # Eğer daha önce bu katman için ağırlık girildiyse, diyalogda göster
        if self.katman_girdileri_widgetlari[katman_index]['custom_weights'] is not None:
            weights_str_list = [" , ".join(map(str, row)) for row in self.katman_girdileri_widgetlari[katman_index]['custom_weights']]
            dialog.weights_input.setPlainText("\n".join(weights_str_list))
        if self.katman_girdileri_widgetlari[katman_index]['custom_biases'] is not None:
            dialog.biases_input.setText(" , ".join(map(str, self.katman_girdileri_widgetlari[katman_index]['custom_biases'].flatten())))


        if dialog.exec(): # exec() diyalogu modal olarak açar ve OK/Cancel beklenir
            weights, biases, error_message = dialog.get_values()
            if error_message: # accept içinde zaten uyarı veriliyor ama çift kontrol
                # QMessageBox.warning(self, "Giriş Hatası", error_message) # Zaten dialog içinde handle ediliyor
                logger.error(f"Ağırlık/Bias diyalogunda hata: {error_message}")
            else:
                self.katman_girdileri_widgetlari[katman_index]['custom_weights'] = weights
                self.katman_girdileri_widgetlari[katman_index]['custom_biases'] = biases
                logger.info(f"Katman {katman_index + 1} için özel ağırlıklar ve biaslar ayarlandı.")
                self.cikti_alani.append(f"Katman {katman_index + 1} için özel ağırlık/bias değerleri kaydedildi.")
                # Butonun metnini veya görünümünü değiştirebiliriz (örn: "Ağırlıklar Girildi ✓")
                self.katman_girdileri_widgetlari[katman_index]['btn_agirlik'].setText("Ağırlıklar ✓")
                self.katman_girdileri_widgetlari[katman_index]['btn_agirlik'].setStyleSheet("color: green;")

        else:
            logger.info(f"Katman {katman_index + 1} için ağırlık/bias ayarı iptal edildi.")

    # Bu fonksiyonla ana arayüzü oluşturuyorum
    def _create_ui(self):

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Başlık bölümü

        # TODO: (Boyutunu büyüteceğim)
        header_layout = QHBoxLayout() 
        main_title_label = QLabel("Neural Network Generator")
        main_title_label.setObjectName("H1Label") 
        main_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.help_button = QToolButton()
        help_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        # help_icon = QIcon("img/help_icon.png")
        
        self.help_button.setIcon(help_icon)
        self.help_button.setIconSize(QSize(24, 24))
        self.help_button.setToolTip("Yardım ve Kullanım Bilgileri")
        self.help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_button.setStyleSheet("QToolButton { border: none; padding: 0px; }") 
        self.help_button.clicked.connect(self.yardim_penceresi_goster_slot) 
        header_layout.addStretch() 
        header_layout.addWidget(main_title_label)
        header_layout.addSpacing(10) 
        header_layout.addWidget(self.help_button)
        header_layout.addStretch()
        self.main_layout.addLayout(header_layout)

        # Başlık bölümü bitti genel parametreler bölümü
        group_genel_parametreler = QGroupBox("Genel Parametreler")

        # FLAG : LR ve EPOCH burada ayarladım.
        ana_layout_genel_param = QHBoxLayout()
        layout_sol_sutun = QFormLayout()
        self.spin_LR = QDoubleSpinBox()
        self.spin_LR.setDecimals(2) 
        self.spin_LR.setMinimum(0.001)
        self.spin_LR.setMaximum(1.0)
        self.spin_LR.setSingleStep(0.01)
        self.spin_LR.setValue(0.1)
        self.spin_LR.setToolTip("Learning Rate")
        layout_sol_sutun.addRow("Öğrenme Oranı:", self.spin_LR)
        self.epoch = QSpinBox()
        self.epoch.setMinimum(1)
        self.epoch.setMaximum(500) 
        self.epoch.setValue(10) 
        self.epoch.setToolTip("Epoch")
        layout_sol_sutun.addRow("Epoch Sayısı:", self.epoch)
        self.spin_input_features = QSpinBox()
        self.spin_input_features.setMinimum(1)
        self.spin_input_features.setMaximum(1024) 
        self.spin_input_features.setValue(2)      
        self.spin_input_features.setToolTip("Ağın girdi katmanındaki özellik (nöron) sayısı.")
        layout_sol_sutun.addRow("Girdi Özellik Sayısı:", self.spin_input_features)

        self.line_edit_input_vector = QLineEdit()
        self.line_edit_input_vector.setPlaceholderText("Örn: 0.5, -0.2 (virgülle ayırın)")
        self.line_edit_input_vector.setToolTip("Test için örnek bir girdi vektörü (virgülle ayrılmış sayılar).")
        layout_sol_sutun.addRow("Örnek Girdi Vektörü:", self.line_edit_input_vector)
    
        self.line_edit_target_vector = QLineEdit()
        self.line_edit_target_vector.setPlaceholderText("Örn: 0.8 (çıktı nöron sayısına göre)")
        self.line_edit_target_vector.setToolTip("Örnek girdi vektörüne karşılık gelen hedef çıktı.")

        layout_sag_sutun = QFormLayout()
        self.combo_loss = QComboBox()
        self.combo_loss.addItems(["Mean Squared Error", "Mean Absolute Error"])
        self.combo_loss.setToolTip("Kayıp Fonksiyonu")
        layout_sag_sutun.addRow("Kayıp Fonksiyonu:", self.combo_loss)
        self.spin_katman_sayisi = QSpinBox()
        self.spin_katman_sayisi.setMinimum(1)
        self.spin_katman_sayisi.setMaximum(20)
        self.spin_katman_sayisi.setValue(2)
        self.spin_katman_sayisi.setToolTip("Toplam Katman Sayısı")
        self.spin_katman_sayisi.valueChanged.connect(self.katman_sayisi_degisti)
        layout_sag_sutun.addRow("Toplam Katman Sayısı:", self.spin_katman_sayisi)

        self.combo_ogrenme_sekli = QComboBox()
        self.combo_ogrenme_sekli.addItems([
            "Batch Gradient Descent",
            "Mini-batch Gradient Descent",
            "Stochastic Gradient Descent (SGD)"
        ])
        self.combo_ogrenme_sekli.setToolTip("Ağırlık güncelleme şekli.")
        # self.combo_ogrenme_sekli.currentIndexChanged.connect(self.ogrenme_sekli_degisti_slot) # Mini-batch için batch size göstermek için
        layout_sag_sutun.addRow("Öğrenme Şekli:", self.combo_ogrenme_sekli)

        # Batch Boyutu (Mini-batch seçilirse görünür olacak)
        self.label_batch_boyutu = QLabel("Batch Boyutu:")
        self.spin_batch_boyutu = QSpinBox()
        self.spin_batch_boyutu.setMinimum(1)
        self.spin_batch_boyutu.setMaximum(1024)
        self.spin_batch_boyutu.setValue(32)   
        self.spin_batch_boyutu.setToolTip("Mini-batch Gradient Descent için batch boyutu. (Minimum 1- Maximum 1024- Default 32 olabilir.)")
        self.label_batch_boyutu.setVisible(False)
        self.spin_batch_boyutu.setVisible(False)

        layout_sag_sutun.addRow(self.label_batch_boyutu, self.spin_batch_boyutu)
        layout_sag_sutun.addRow("Örnek Hedef Çıktı (Input vector icin):", self.line_edit_target_vector)
        self.combo_ogrenme_sekli.currentIndexChanged.connect(self.ogrenme_sekli_degisti_slot)

        ana_layout_genel_param.addLayout(layout_sol_sutun)
        ana_layout_genel_param.addSpacing(20)
        ana_layout_genel_param.addLayout(layout_sag_sutun)
        group_genel_parametreler.setLayout(ana_layout_genel_param) 
        self.main_layout.addWidget(group_genel_parametreler)

        #  Katman Detayları bölümü
    
        self.label_kaydirma_uyarisi = QLabel("(Tüm katmanları görmek için aşağı kaydırın)")
        self.label_kaydirma_uyarisi.setStyleSheet("font-style: italic; color: gray;") 
        self.label_kaydirma_uyarisi.setVisible(False)
        self.group_katman_detaylari = QGroupBox("Katman Detayları")
        self.layout_katman_detaylari = QVBoxLayout()
        self.group_katman_detaylari.setLayout(self.layout_katman_detaylari)

        self.scroll_area_katman_detaylari = QScrollArea() 
        self.scroll_area_katman_detaylari.setWidgetResizable(True) 
        self.scroll_area_katman_detaylari.setWidget(self.group_katman_detaylari)
        self.main_layout.addWidget(self.scroll_area_katman_detaylari)
        self.katman_sayisi_degisti(self.spin_katman_sayisi.value())

        ## split

        # İslemler (Butonları)
        islemler = QGroupBox("İşlemler")
        layout_islemler = QHBoxLayout()

        self.reset_model_parameters = QPushButton("Model Parametrelerini Sıfırla")
        self.iterate_over_network = QPushButton("Modeli Eğit (Iteratif)")
        self.generate_model_train = QPushButton("Modeli Oluştur ve Eğit")

        self.reset_model_parameters.setToolTip("Tüm giriş alanlarını varsayılan değerlerine döndürür.")
        self.reset_model_parameters.clicked.connect(self.parametreleri_sifirla_slot) 

        self.iterate_over_network.setToolTip("Modeli adım adım eğitir ve ara çıktıları (loss değerleri gibi) gösterir.")
        self.iterate_over_network.clicked.connect(self.iteratif_egitim_slot) 

        self.generate_model_train.setToolTip("Girilen parametrelerle sinir ağını oluşturur ve tüm epoch'lar için eğitimi başlatır.")
        self.generate_model_train.clicked.connect(self.generate_and_train_model_slot) # Bağlantı

        # self.btn_modeli_olustur.clicked.connect(self.model_parametrelerini_oku_slot) # Bu daha sonra eklenecek
        layout_islemler.addWidget(self.generate_model_train)
        layout_islemler.addWidget(self.iterate_over_network)
        layout_islemler.addWidget(self.reset_model_parameters)
        islemler.setLayout(layout_islemler)
        self.main_layout.addWidget(islemler)

        # split

        self.group_cikti_alani = QGroupBox("Çıktı ve Loglar")
        layout_cikti = QVBoxLayout()
        self.cikti_alani = QTextEdit()
        self.cikti_alani.setReadOnly(True)
        self.cikti_alani.setPlaceholderText("Ağ oluşturulduğunda veya eğitildiğinde sonuçlar burada görünecektir...")
        self.cikti_alani.setMinimumHeight(150)
        layout_cikti.addWidget(self.cikti_alani)
        self.group_cikti_alani.setLayout(layout_cikti)
        self.main_layout.addWidget(self.group_cikti_alani)
        self.main_layout.addStretch(1) 

        # split 

        self._create_menu_bar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("NNG çalışıyor...")

    def _create_menu_bar(self):

        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("&Yardım")
        help_content_action = QAction("Yardım İçeriği", self)
        help_content_action.setStatusTip("Uygulamanın nasıl kullanılacağına dair bilgi.")
        help_content_action.triggered.connect(self.yardim_penceresi_goster_slot) 
        help_menu.addAction(help_content_action)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        app.setStyleSheet(dark_stylesheet)
    except NameError:
        print("Uyarı: 'dark_stylesheet' bulunamadı. Lütfen stylesheets.py dosyasını kontrol edin.")
    except Exception as e:
        print(f"Stil sayfası yüklenirken bir hata oluştu: {e}")

    window = GeneratorWindow()
    window.show()
    sys.exit(app.exec())