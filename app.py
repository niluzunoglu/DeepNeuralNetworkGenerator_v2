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
            spin_noron_sayisi.setMaximum(1024) # Makul bir üst sınır
            spin_noron_sayisi.setValue(10) # Varsayılan nöron sayısı
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

            self.layout_katman_detaylari.addLayout(katman_layout)
            self.katman_girdileri_widgetlari.append({
                'noron_spinbox': spin_noron_sayisi,
                'aktivasyon_combobox': combo_aktivasyon
            })

    def parametreleri_sifirla_slot(self):

        logger.info("'Parametreleri Sıfırla' butonuna tıklandı.")
        self.varsayilan_degerleri_ayarla()
        # Katman detayları da varsayılan katman sayısına göre güncellenmeli
        self.katman_sayisi_degisti(self.spin_katman_sayisi.value())
        self.cikti_alani.setText("Parametreler varsayılan değerlere sıfırlandı.\nYeni bir ağ tanımlayabilirsiniz.")
        self.status_bar.showMessage("Parametreler sıfırlandı.")

    def varsayilan_degerleri_ayarla(self):

        if hasattr(self, 'spin_LR'): self.spin_LR.setValue(0.01)
        if hasattr(self, 'epoch'): self.epoch.setValue(1000)
        if hasattr(self, 'combo_loss'): self.combo_loss.setCurrentIndex(0) # İlk elemana ayarlasın MSE .
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
        
        # === AŞAMA 3'TE BURAYA GERÇEK AĞ OLUŞTURMA VE EĞİTİM KODU GELECEK ===
        # Örnek:
        # try:
        #     self.network_instance = Network() # implementations.Network
        #     input_size_onceki_katman = X_train.shape[1] # Girdi verisinin özellik sayısı
        #     for i, config in enumerate(katman_yapilandirmalari):
        #         # Aktivasyon string'ini gerçek sınıfa dönüştür
        #         activation_class = self.get_activation_class_from_string(config['aktivasyon'])
        #         self.network_instance.add_layer(
        #             DenseLayer(input_size_onceki_katman, config['noron'], activation_function=activation_class(), name=f"Katman_{i+1}")
        #         )
        #         input_size_onceki_katman = config['noron']
            
        #     loss_class = self.get_loss_class_from_string(secilen_loss_fonksiyonu_str)
            
        #     # Eğitimi ayrı bir thread'de başlatmak iyi olur GUI'nin donmaması için
        #     self.cikti_alani.append("\nEğitim başlıyor...")
        #     # history = self.network_instance.train(X_train, y_train, epoch_sayisi, ogrenme_orani, loss_class())
        #     # self.cikti_alani.append("\nEğitim tamamlandı.")
        #     # self.cikti_alani.append(f"Son Kayıp: {history['loss'][-1]}")
        # except Exception as e:
        #     logger.error(f"Ağ oluşturma veya eğitim sırasında hata: {e}", exc_info=True)
        #     self.cikti_alani.append(f"\nHATA: Ağ oluşturma veya eğitim sırasında bir sorun oluştu.\nDetaylar için loglara bakın.\n{e}")
        # =========================================================================
        self.status_bar.showMessage("Ağ parametreleri okundu. Eğitim için hazır (Aşama 3).")

    def iteratif_egitim_slot(self):
        logger.info("'Modeli Eğit (İteratif)' butonuna tıklandı.")
        self.cikti_alani.append("\n" + "="*50 + "\nİTERATİF EĞİTİM MODU (Henüz implemente edilmedi)\n" + "="*50)
        self.cikti_alani.append("Bu modda, ağ adım adım (belki her epoch'ta bir) eğitilecek ve sonuçlar güncellenecektir.")
        self.status_bar.showMessage("İteratif eğitim modu seçildi (henüz aktif değil).")
        
    def yardim_penceresi_goster_slot(self):

        logger.info("Yardım penceresi gösteriliyor.")
        yardim_basligi = "Neural Network Generator - Yardım"
        yardim_metni = """
        <h2>Neural Network Generator</h2>
        <p>Bu simülator bir neural network oluşturmayı ve bu network üzerinde temel parametre ayarlarını yapmanızı sağlar.</p>
        <p>Arayüzde görünen temel parametrelerin anlamları ve kullanım sınırları aşağıdaki gibidir: </p>

        <h3>Genel Parametreler Penceresi:</h3>
        <ul>
            <li><b>Öğrenme Oranı:</b> Default değeri 0.01'dir. Her bir artırımda 0.01 artar. En fazla 1 olabilir.</li>
            <li><b>Epoch Sayısı:</b> Tüm eğitim veri setinin networkten kaç kez geçirileceğini belirtir. En fazla 500 olabilir. </li>
            <li><b>Kayıp Fonksiyonu:</b> Bulunan sonucun gerçek sonuçtan ne kadar farklı olduğunun ölçüsüdür. MSE, RMSE gibi değerler alabilir. (örn: Mean Squared Error).</li>
            <li><b>Toplam Katman Sayısı:</b> Networkteki katmanların toplam sayısını belirtir. (girdi katmanı hariç, çıktı katmanı dahil). En fazla 20 olabilir.</li>
        </ul>

        <h3>Katman Detayları:</h3>
        <p>"Toplam Katman Sayısı" değiştirildiğinde, her katman için aşağıdaki ayarlar yapılabilir:</p>
        <p> Katman sayısı değiştikçe katman eklemek için yeni pencereler oluşacaktır, katman detayları penceresini aşağı kaydırarak görebilirsiniz.</p>
        <ul>
            <li><b>Nöron Sayısı:</b> O katmanda bulunacak nöron sayısı.</li>
            <li><b>Aktivasyon Fonksiyonu:</b> ReLU, Sigmoid, Tanh gibi aktivasyon fonksiyonu seçilebilir.</li>
        </ul>

        <h3>İşlemler:</h3>
        <ul>
            <li><b>Modeli Oluştur ve Eğit:</b> Girilen parametrelerle ağı oluşturur ve belirlenen öğrenme şekliyle (MBGD, BGD, SGD) tüm epoch'lar için eğitimi başlatır.</li>
            <li><b>Modeli Eğit (İteratif):</b> Ağı adım adım eğitmenizi sağlar. Burada her bir eğitimde forward ve backward propagation işlemlerini gözlemleyebilirsiniz.</li>
            <li><b>Model Parametrelerini Sıfırla:</b> Tüm giriş alanlarını varsayılan değerlerine döndürür.</li>
        </ul>
        
        <p>Daha fazla bilgi veya sorunlarınız için nil.uzunoglu@std.yildiz.edu.tr adresinden iletişime geçebilirsiniz.</p>
        <hr>
        <p><i>Versiyon: 2.0 </i></p>
        """
        
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
        self.spin_katman_sayisi.setToolTip("Toplam Yoğun Katman Sayısı")
        self.spin_katman_sayisi.valueChanged.connect(self.katman_sayisi_degisti)
        layout_sag_sutun.addRow("Toplam Katman Sayısı:", self.spin_katman_sayisi)

        self.combo_ogrenme_sekli = QComboBox()
        self.combo_ogrenme_sekli.addItems([
            "Batch Gradient Descent",
            "Mini-batch Gradient Descent",
            "Stochastic Gradient Descent (SGD)"
        ])
        self.combo_ogrenme_sekli.setToolTip("Ağırlık güncelleme stratejisi.")
        # self.combo_ogrenme_sekli.currentIndexChanged.connect(self.ogrenme_sekli_degisti_slot) # Mini-batch için batch size göstermek için
        layout_sag_sutun.addRow("Öğrenme Şekli:", self.combo_ogrenme_sekli)

        # Batch Boyutu (Mini-batch seçilirse görünür olacak)
        self.label_batch_boyutu = QLabel("Batch Boyutu:")
        self.spin_batch_boyutu = QSpinBox()
        self.spin_batch_boyutu.setMinimum(1)
        self.spin_batch_boyutu.setMaximum(1024)
        self.spin_batch_boyutu.setValue(32)   
        self.spin_batch_boyutu.setToolTip("Mini-batch Gradient Descent için batch boyutu.")

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