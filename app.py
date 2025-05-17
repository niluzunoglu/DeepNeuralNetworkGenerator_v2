import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStatusBar,
    QMenuBar,
    QPushButton,    # Eklendi
    QLineEdit,      # Eklendi
    QTextEdit,      # Eklendi
    QComboBox,      # Eklendi
    QCheckBox,      # Eklendi
    QRadioButton,   # Eklendi
    QSlider,        # Eklendi
    QProgressBar,   # Eklendi
    QSpinBox,       # Eklendi
    QTabWidget,     # Eklendi
    QGroupBox,      # Eklendi (Widget'ları gruplamak için)
    QFormLayout,    # Eklendi (Tab içinde form düzeni için)
    QHBoxLayout     # Eklendi (Bazı widget'ları yatay sıralamak için)
)
from PyQt6.QtGui import QAction # QPalette ve QColor'a artık burada ihtiyacımız yok
                               # eğer stylesheet her şeyi hallediyorsa.
from PyQt6.QtCore import Qt     # QSlider için eklendi

# Bu satırın çalışması için projenizin ana dizininde
# 'stylesheets.py' adında bir dosya ve içinde 'dark_stylesheet'
# adında bir string değişken olmalıdır.
# Örnek stylesheets.py içeriği:
# dark_stylesheet = """
# QWidget { background-color: #333; color: white; }
# QPushButton { background-color: #555; border: 1px solid #777; }
# /* ... diğer stiller ... */
# """
from stylesheets import dark_stylesheet

class GeneratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Generator")
        self.setGeometry(100, 100, 850, 700) # Biraz daha geniş ve yüksek yaptım
        self._create_ui()

    # Bu fonksiyonla ana arayüzü oluşturuyorum
    def _create_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Başlık bölümü TODO: (Boyutunu büyüteceğim)
        main_title_label = QLabel("Neural Network Generator")
        main_title_label.setObjectName("H1Label") 
        main_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.main_layout.addWidget(main_title_label)

        # --- Grup Kutusu 1: Giriş Alanları ve Butonlar ---
        group_box1 = QGroupBox("Temel Kontroller")
        group_layout1 = QVBoxLayout()

        # Buton
        self.button = QPushButton("Örnek Buton")
        self.button.setObjectName("PrimaryButton")
        self.button.setToolTip("Bu bir QPushButton örneğidir.")
        group_layout1.addWidget(self.button)

        # LineEdit (Tek satır giriş)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Tek satırlık metin girişi...")
        group_layout1.addWidget(self.line_edit)

        # ComboBox (Açılır Liste)
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Seçenek Alpha", "Seçenek Beta", "Pasif Seçenek Gamma", "Seçenek Delta"])
        self.combo_box.model().item(2).setEnabled(False) # Bir öğeyi pasif yapalım
        group_layout1.addWidget(self.combo_box)

        # SpinBox (Sayı girişi)
        self.spin_box = QSpinBox()
        self.spin_box.setRange(-10, 100)
        self.spin_box.setValue(10)
        self.spin_box.setPrefix("Değer: ")
        self.spin_box.setSuffix(" birim")
        group_layout1.addWidget(self.spin_box)

        group_box1.setLayout(group_layout1)
        self.main_layout.addWidget(group_box1)


        # --- Grup Kutusu 2: Seçenekler ve Ayarlar ---
        group_box2 = QGroupBox("Seçim ve Ayar Elemanları")
        group_layout2 = QVBoxLayout()

        # CheckBox'lar (yatayda daha iyi görünmesi için QHBoxLayout içinde)
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_container)
        self.checkbox1 = QCheckBox("Aktif")
        self.checkbox1.setChecked(True)
        self.checkbox2 = QCheckBox("Öncelikli")
        self.checkbox3 = QCheckBox("Devre Dışı")
        self.checkbox3.setEnabled(False) # Pasif CheckBox
        checkbox_layout.addWidget(self.checkbox1)
        checkbox_layout.addWidget(self.checkbox2)
        checkbox_layout.addWidget(self.checkbox3)
        checkbox_layout.addStretch() # Sağa yaslamak için
        group_layout2.addWidget(checkbox_container)


        radio_container = QWidget()
        radio_layout = QHBoxLayout(radio_container)
        self.radio1 = QRadioButton("Mod 1")
        self.radio2 = QRadioButton("Mod 2")
        self.radio3 = QRadioButton("Mod 3 (Pasif)")
        self.radio1.setChecked(True)
        self.radio3.setEnabled(False) # Pasif RadioButton
        radio_layout.addWidget(QLabel("Çalışma Modu:"))
        radio_layout.addWidget(self.radio1)
        radio_layout.addWidget(self.radio2)
        radio_layout.addWidget(self.radio3)
        radio_layout.addStretch()
        group_layout2.addWidget(radio_container)

        # Slider (Kaydırıcı)
        group_layout2.addWidget(QLabel("Hassasiyet Ayarı:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(65)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)
        group_layout2.addWidget(self.slider)

        # ProgressBar (İlerleme Çubuğu)
        group_layout2.addWidget(QLabel("İlerleme Durumu:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(40)
        # self.progress_bar.setTextVisible(False) # İsteğe bağlı: yüzdeyi gizle
        group_layout2.addWidget(self.progress_bar)

        group_box2.setLayout(group_layout2)
        self.main_layout.addWidget(group_box2)


        # --- Sekmeli Arayüz (QTabWidget) ---
        self.tab_widget = QTabWidget()

        # Sekme 1: Metin Alanı
        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Buraya uzun açıklamalarınızı veya kodunuzu yazabilirsiniz...")
        self.text_edit.setPlainText("Örnek bir metin.\nBirden fazla satır içerebilir.\n\nStil sayfanızın nasıl göründüğünü test edin.")
        tab1_layout.addWidget(self.text_edit)
        self.tab_widget.addTab(tab1_widget, "Detaylı Notlar")

        # Sekme 2: Form Düzeni
        tab2_widget = QWidget()
        tab2_layout = QFormLayout(tab2_widget) # Form düzeni için QFormLayout idealdir
        tab2_layout.addRow("Proje Adı:", QLineEdit("NeuroForge Projesi"))
        tab2_layout.addRow("Versiyon:", QLineEdit("0.1 Alpha"))
        spin_katman_sayisi = QSpinBox()
        spin_katman_sayisi.setValue(5)
        tab2_layout.addRow("Katman Sayısı:", spin_katman_sayisi)
        tab2_layout.addRow("Kaydet:", QPushButton("Ayarları Kaydet"))
        self.tab_widget.addTab(tab2_widget, "Proje Bilgileri")

        # Sekme 3: Pasif Sekme
        tab3_widget = QWidget()
        tab3_layout = QVBoxLayout(tab3_widget)
        tab3_layout.addWidget(QLabel("Bu sekme şu anda kullanım dışıdır."))
        self.tab_widget.addTab(tab3_widget, "Gelişmiş Ayarlar")
        self.tab_widget.setTabEnabled(2, False) # Bu sekmeyi pasif yapalım

        self.main_layout.addWidget(self.tab_widget)

        # --- Ana Layout'un sonuna boşluk ekleyerek widget'ları yukarı itme (isteğe bağlı ama şık durur) ---
        self.main_layout.addStretch(1)

        # --- Menü Çubuğu ---
        self._create_menu_bar()

        # --- Durum Çubuğu ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Arayüz yüklendi, tema aktif.")


    def _create_menu_bar(self):
        """Menü çubuğunu ve menüleri oluşturur."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&Dosya")
        new_action = QAction("&Yeni Model", self)
        new_action.setStatusTip("Yeni bir sinir ağı modeli oluştur")
        file_menu.addAction(new_action)
        open_action = QAction("&Model Aç", self)
        open_action.setStatusTip("Var olan bir model dosyasını aç")
        file_menu.addAction(open_action)
        save_action = QAction("&Modeli Kaydet", self)
        save_action.setStatusTip("Mevcut modeli kaydet")
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction("&Çıkış", self)
        exit_action.setStatusTip("Uygulamadan çık")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("&Düzenle")
        undo_action = QAction("&Geri Al", self)
        undo_action.setStatusTip("Son işlemi geri al")
        undo_action.setEnabled(False) # Örnek olarak pasif
        edit_menu.addAction(undo_action)
        redo_action = QAction("&İleri Al", self)
        redo_action.setStatusTip("Geri alınan işlemi yinele")
        redo_action.setEnabled(False)
        edit_menu.addAction(redo_action)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Daha modern bir temel görünüm için

    # dark_stylesheet'in stylesheets.py dosyasından yüklendiğini varsayıyoruz
    try:
        app.setStyleSheet(dark_stylesheet)
    except NameError:
        print("Uyarı: 'dark_stylesheet' bulunamadı. Lütfen stylesheets.py dosyasını kontrol edin.")
    except Exception as e:
        print(f"Stil sayfası yüklenirken bir hata oluştu: {e}")


    window = GeneratorWindow()
    window.show()
    sys.exit(app.exec())