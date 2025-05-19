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

        if hasattr(self, 'spin_LR'): self.spin_LR.setValue(0.01)
        if hasattr(self, 'epoch'): self.epoch.setValue(1000)
        if hasattr(self, 'combo_loss'): self.combo_loss.setCurrentIndex(0) 
        if hasattr(self, 'spin_katman_sayisi'): self.spin_katman_sayisi.setValue(2) 

        logger.info("Giriş alanları varsayılan değerlere ayarlandı.")

    def generate_and_train_model_slot(self):

        logger.info("'Modeli Oluştur ve Eğit' butonuna tıklandı.")
        self.cikti_alani.clear() 
        self.cikti_alani.setText("Ağ parametreleri okunuyor ve ağ oluşturuluyor...\nLütfen bekleyin...\n" + "="*50)

        ogrenme_orani = self.spin_LR.value()
        epoch_sayisi = self.epoch.value()
        secilen_loss_fonksiyonu_str = self.combo_loss.currentText()
        toplam_katman_sayisi = self.spin_katman_sayisi.value()

        self.cikti_alani.append(f"\nGenel Parametreler:")
        self.cikti_alani.append(f"  Öğrenme Oranı: {ogrenme_orani}")
        self.cikti_alani.append(f"  Epoch Sayısı: {epoch_sayisi}")
        self.cikti_alani.append(f"  Kayıp Fonksiyonu: {secilen_loss_fonksiyonu_str}")
        self.cikti_alani.append(f"  Toplam Katman Sayısı: {toplam_katman_sayisi}")

        self.cikti_alani.append("\nKatman Detayları:")
        katman_yapilandirmalari = []
        for i, katman_widget_grubu in enumerate(self.katman_girdileri_widgetlari):
            katman_no = i + 1
            noron_sayisi = katman_widget_grubu['noron_spinbox'].value()
            aktivasyon_str = katman_widget_grubu['aktivasyon_combobox'].currentText()
            self.cikti_alani.append(f"  Katman {katman_no}: Nöron Sayısı={noron_sayisi}, Aktivasyon={aktivasyon_str}")
            katman_yapilandirmalari.append({
                'noron': noron_sayisi,
                'aktivasyon': aktivasyon_str
            })
        
        self.cikti_alani.append("="*50 + "\nParametreler başarıyla okundu.")
        self.cikti_alani.append("Şu anda bu parametrelerle bir ağ oluşturulacak ve eğitim başlatılacak (Aşama 3).")
        
        self.cikti_alani.append("\n" + "="*10 + " AĞ OLUŞTURULUYOR (TASLAK) " + "="*10)
        try:
            self.network_instance = Network() 
            
            # TODO: İlk katmanın input_size'ını belirle (UI'dan veya veri setinden)
            # Şimdilik kullanıcıdan almadığımız için sabit bir değer veya hata verelim.
            # Örneğin, Genel Parametreler'e bir "Girdi Özellik Sayısı" SpinBox'ı eklenebilir.
            # Veya XOR gibi bilinen bir problem için sabitlenebilir.

            input_feature_count = 2 # Örnek olarak XOR için 2
            self.cikti_alani.append(f"UYARI: Girdi katmanı özellik sayısı varsayılan olarak {input_feature_count} alındı.")
            
            input_size_onceki_katman = input_feature_count

            for i, katman_config_ui in enumerate(katman_yapilandirmalari): # katman_yapilandirmalari UI'dan okunanlar
                # Aktivasyon string'ini gerçek sınıfa dönüştür
                # activation_class = self.get_activation_class_from_string(katman_config_ui['aktivasyon']) # Bu yardımcı fonksiyonu yazmanız gerekebilir
                # Şimdilik basit if/else ile:
                if katman_config_ui['aktivasyon'] == "ReLU": activation_instance = ReLU()
                elif katman_config_ui['aktivasyon'] == "Sigmoid": activation_instance = Sigmoid()
                elif katman_config_ui['aktivasyon'] == "Tanh": activation_instance = Tanh()
                else: activation_instance = Linear()

                logger.info(f"Katman {i+1} oluşturuluyor: input_size={input_size_onceki_katman}, output_size={katman_config_ui['noron']}")
                
                new_layer = DenseLayer(
                    input_size_onceki_katman,
                    katman_config_ui['noron'],
                    activation_function=activation_instance,
                    name=f"Katman_{i+1}"
                )

                custom_w = self.katman_girdileri_widgetlari[i]['custom_weights']
                custom_b = self.katman_girdileri_widgetlari[i]['custom_biases']

                if custom_w is not None:
                    if new_layer.weights.shape == custom_w.shape:
                        new_layer.weights = custom_w
                        self.cikti_alani.append(f"  Katman {i+1} için özel ağırlıklar kullanıldı.")
                    else:
                        self.cikti_alani.append(f"  UYARI: Katman {i+1} için girilen özel ağırlıkların boyutu ({custom_w.shape}) katman boyutuyla ({new_layer.weights.shape}) eşleşmiyor! Rastgele ağırlıklar kullanılacak.")
                
                if custom_b is not None:
                    if new_layer.biases.shape == custom_b.shape:
                        new_layer.biases = custom_b
                        self.cikti_alani.append(f"  Katman {i+1} için özel biaslar kullanıldı.")
                    else:
                        self.cikti_alani.append(f"  UYARI: Katman {i+1} için girilen özel biasların boyutu ({custom_b.shape}) katman boyutuyla ({new_layer.biases.shape}) eşleşmiyor! Rastgele biaslar kullanılacak.")

                self.network_instance.add_layer(new_layer)
                input_size_onceki_katman = katman_config_ui['noron']
            
            self.cikti_alani.append(f"Ağ başarıyla oluşturuldu:\n{self.network_instance}")
            # loss_class = self.get_loss_class_from_string(secilen_loss_fonksiyonu_str)
            # ... (eğitim kısmı) ...

        except Exception as e:
            logger.error(f"Ağ oluşturma sırasında hata: {e}", exc_info=True)
            self.cikti_alani.append(f"\nHATA: Ağ oluşturma sırasında bir sorun oluştu.\nDetaylar için loglara bakın.\n{e}")

        self.status_bar.showMessage("Ağ parametreleri okundu. Ağ oluşturuldu")

    def iteratif_egitim_slot(self):
        logger.info("'Modeli Eğit (İteratif)' butonuna tıklandı.")
        self.cikti_alani.append("\n" + "="*50 + "\nİTERATİF EĞİTİM MODU (Henüz implemente edilmedi)\n" + "="*50)
        self.cikti_alani.append("Bu modda, ağ adım adım (belki her epoch'ta bir) eğitilecek ve sonuçlar güncellenecektir.")
        self.status_bar.showMessage("İteratif eğitim modu seçildi (henüz aktif değil).")
        
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
        
        # Diyalogu açmadan önce, bu katmanın ve bir önceki katmanın nöron sayılarını bilmemiz gerek.
        # Bir önceki katmanın nöron sayısı:
        # Eğer bu ilk katman ise (katman_index == 0), input_size UI'dan alınmalı (henüz yok, şimdilik sabit varsayalım)
        # TODO: Gerçek input size'ı belirlemek için bir mekanizma lazım (örn: veri seti yüklendiğinde)
        # Şimdilik varsayılan bir girdi boyutu kullanalım veya kullanıcıya soralım.
        # Basitlik adına, eğer ilk katmansa prev_layer_neurons için bir varsayım yapalım.
        # Ya da daha iyisi, bu bilgiyi `katman_girdileri_widgetlari` içinde saklayalım.
        # Şimdilik, ilk katman için input_size'ı UI'da olmayan bir yerden almamız gerekecek
        # veya bu özelliği sadece >1 katman için aktif edebiliriz.

        # Basit bir varsayım: İlk katmanın input_size'ı UI'da bir "Giriş Nöron Sayısı" spinbox'ından alınacak.
        # O spinbox henüz olmadığı için, şimdilik sabit bir değer veya hata verelim.
        # Bu, UI'da "Giriş Katmanı Nöron Sayısı" gibi bir alan eklemeyi gerektirebilir.
        # Şimdilik bunu bir TODO olarak bırakıp, test için sabit bir değer kullanalım.

        if katman_index == 0:
            # TODO: UI'dan gerçek girdi özellik sayısını al. Şimdilik sabit bir değer.
            prev_layer_neurons = getattr(self, 'input_feature_count', 2) # Varsayılan 2 özellik
            logger.warning(f"İlk katman için girdi nöron sayısı varsayılan olarak {prev_layer_neurons} alındı. UI'dan alınmalı.")
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
        # Sinyal bağlantısı (combo_ogrenme_sekli değiştiğinde batch boyutu widget'larını göster/gizle)
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