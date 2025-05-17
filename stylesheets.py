# stylesheets.py

# --- RENK PALETİ (Modern ve Sakin Koyu Tema) ---
# Zemin Renkleri
APP_BACKGROUND = "#2B303B"
CONTENT_BACKGROUND = "#343A46"
INPUT_BACKGROUND = "#272B33" # Giriş alanları için zeminden biraz daha koyu
HOVER_BACKGROUND = "#3F4552"
ACTIVE_SELECTION_BACKGROUND = "#4A5160" # Seçili ama odaklı olmayan öğeler için

# Metin Renkleri
TEXT_PRIMARY = "#D8DEE9"
TEXT_SECONDARY = "#A3ABB9"
TEXT_ON_ACCENT = "#FFFFFF"
TEXT_PLACEHOLDER = "#677085" # Placeholder için özel renk

# Vurgu Renkleri
ACCENT_PRIMARY = "#5E81AC"
ACCENT_PRIMARY_HOVER = "#6B95C9"
ACCENT_PRIMARY_PRESSED = "#55759A" # Basıldığında
ACCENT_SECONDARY = "#88C0D0"
ACCENT_SUCCESS = "#A3BE8C"
ACCENT_WARNING = "#EBCB8B"
ACCENT_DANGER = "#BF616A"

# Kenarlıklar
BORDER_COLOR = "#4C566A"
BORDER_FOCUSED = ACCENT_PRIMARY
BORDER_INPUT = "#434A58" # Giriş alanları için normal kenarlık

dark_stylesheet = f"""
/* === GENEL AYARLAR === */
QWidget {{
    background-color: {CONTENT_BACKGROUND};
    color: {TEXT_PRIMARY};
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif; /* Modern fontlar */
    font-size: 12pt;
    border: none;
}}

QMainWindow {{
    background-color: {APP_BACKGROUND};
}}

/* === TEMEL WIDGET'LAR === */
QLabel {{
    background-color: transparent;
    padding: 1px; /* Padding'i minimuma indir, layout ile ayarla */
}}
QLabel#H1Label {{ font-size: 15pt; font-weight: 600; color: {TEXT_PRIMARY}; padding-bottom: 8px; }}
QLabel#H2Label {{ font-size: 12pt; font-weight: 500; color: {TEXT_PRIMARY}; padding-bottom: 5px; }}
QLabel#ErrorLabel {{ color: {ACCENT_DANGER}; }}
QLabel#SuccessLabel {{ color: {ACCENT_SUCCESS}; }}

QPushButton {{
    background-color: {INPUT_BACKGROUND}; /* Butonlar giriş alanlarıyla aynı zeminde */
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_INPUT};
    padding: 7px 15px;
    border-radius: 4px;
    min-height: 24px;
    font-weight: 500; /* Orta kalınlık */
}}
QPushButton:hover {{
    background-color: {HOVER_BACKGROUND};
    border-color: {BORDER_COLOR};
}}
QPushButton:pressed {{
    background-color: {ACTIVE_SELECTION_BACKGROUND};
    border-color: {BORDER_FOCUSED};
}}
QPushButton:focus {{
    outline: none;
    border: 1px solid {BORDER_FOCUSED};
    /* padding: 6px 14px; /* Kenarlık için gerekirse ayarla */
}}
QPushButton:disabled {{
    background-color: {CONTENT_BACKGROUND};
    color: {TEXT_SECONDARY};
    border-color: {CONTENT_BACKGROUND};
}}

/* Özel Buton Tipleri (setObjectName ile kullanılır) */
QPushButton#PrimaryButton {{
    background-color: {ACCENT_PRIMARY};
    color: {TEXT_ON_ACCENT};
    border-color: {ACCENT_PRIMARY};
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover {{ background-color: {ACCENT_PRIMARY_HOVER}; border-color: {ACCENT_PRIMARY_HOVER}; }}
QPushButton#PrimaryButton:pressed {{ background-color: {ACCENT_PRIMARY_PRESSED}; border-color: {ACCENT_PRIMARY_PRESSED}; }}

QPushButton#SecondaryButton {{
    background-color: transparent;
    color: {ACCENT_SECONDARY};
    border: 1px solid {ACCENT_SECONDARY};
}}
QPushButton#SecondaryButton:hover {{ background-color: rgba(136, 192, 208, 0.1); }} /* %10 opaklık */
QPushButton#SecondaryButton:pressed {{ background-color: rgba(136, 192, 208, 0.2); }}

QPushButton#DangerButton {{
    background-color: {ACCENT_DANGER};
    color: {TEXT_ON_ACCENT};
    border-color: {ACCENT_DANGER};
}}
QPushButton#DangerButton:hover {{ background-color: #D08770; border-color: #D08770; }} /* Nord'un kırmızısının açığı */

/* === GİRİŞ ALANLARI === */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {INPUT_BACKGROUND};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_INPUT};
    border-radius: 4px;
    padding: 7px;
    min-height: 24px;
    selection-background-color: {ACCENT_PRIMARY}; /* Seçili metin arka planı */
    selection-color: {TEXT_ON_ACCENT};      /* Seçili metin rengi */
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BORDER_FOCUSED};
    background-color: {INPUT_BACKGROUND}; /* Odaklandığında arka plan değişmeyebilir, sadece kenarlık */
}}
QLineEdit:read-only, QTextEdit:read-only, QPlainTextEdit:read-only {{
    background-color: {CONTENT_BACKGROUND};
    color: {TEXT_SECONDARY};
    border-color: {CONTENT_BACKGROUND};
}}
/* Placeholder text color - QPalette ile daha güvenilir olabilir */
QLineEdit[placeholderText] {{ color: {TEXT_PLACEHOLDER}; }}


/* === QComboBox === */
QComboBox {{
    background-color: {INPUT_BACKGROUND};
    border: 1px solid {BORDER_INPUT};
    border-radius: 4px;
    padding: 1px 1px 1px 7px; /* Sol padding metin için, sağ taraf ok için ayarlanacak */
    min-height: 24px;
    font-weight: 500;
}}
QComboBox:focus {{ border-color: {BORDER_FOCUSED}; }}
QComboBox:on {{ /* Açıldığında */
    padding-top: 2px; /* İçerik ve çerçeve arasında hafif bir sınır */
    padding-left: 8px;
    /* background-color: {ACTIVE_SELECTION_BACKGROUND}; */
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left-width: 1px;
    border-left-color: {BORDER_INPUT};
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}}
QComboBox::down-arrow {{
    /* Modern bir ok için SVG veya font ikonu tercih edilir. */
    /* Basit bir QSS üçgeni: */
    border-style: solid;
    border-width: 4px 4px 0 4px; /* Üçgen boyutları */
    border-color: {TEXT_SECONDARY} transparent transparent transparent;
    width: 0; height: 0;
    margin: auto 5px; /* Ortalama */
}}
QComboBox::down-arrow:on {{
    border-top-color: {ACCENT_PRIMARY}; /* Açıkken okun rengi değişebilir */
    /* top: 1px; /* Hafif aşağı kaydırma */
}}
QComboBox QAbstractItemView {{ /* Açılır liste */
    background-color: {INPUT_BACKGROUND};
    border: 1px solid {BORDER_FOCUSED};
    border-radius: 4px;
    selection-background-color: {ACCENT_PRIMARY};
    selection-color: {TEXT_ON_ACCENT};
    padding: 4px;
    outline: 0px; /* İstenmeyen çerçeveyi kaldır */
    margin-top: 2px; /* ComboBox ile liste arasında boşluk */
}}

/* === SEÇİM KUTULARI VE RADYO BUTONLARI === */
QCheckBox, QRadioButton {{
    spacing: 8px;
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-weight: 500;
    height: 22px; /* Dikey hizalama için */
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    background-color: {INPUT_BACKGROUND};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT_PRIMARY}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT_PRIMARY};
    border-color: {ACCENT_PRIMARY};
    /* image: url(:/icons/checkmark.svg); /* SVG checkmark */
}}
QCheckBox::indicator:checked:hover, QRadioButton::indicator:checked:hover {{
    background-color: {ACCENT_PRIMARY_HOVER};
    border-color: {ACCENT_PRIMARY_HOVER};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {TEXT_SECONDARY}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {CONTENT_BACKGROUND};
    border-color: {CONTENT_BACKGROUND};
}}

/* === KAYDIRICI (SLIDER) === */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {INPUT_BACKGROUND};
    margin: 2px 0;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_PRIMARY};
    border: none;
    width: 16px; height: 16px; /* Genişlik ve yükseklik eşit, tam yuvarlak */
    margin: -5px 0; /* Dikey ortala */
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_PRIMARY_HOVER}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_PRIMARY}; border-radius: 3px; }}
QSlider::add-page:horizontal {{ background: {INPUT_BACKGROUND}; border-radius: 3px; }}

/* === İLERLEME ÇUBUĞU (PROGRESS BAR) === */
QProgressBar {{
    border: 1px solid {BORDER_INPUT};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_ON_ACCENT};
    background-color: {INPUT_BACKGROUND};
    height: 22px;
    font-weight: 500;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_SECONDARY}; /* Farklı bir vurgu rengi */
    border-radius: 3px;
    /* margin: 1px; */
}}

/* === SEKME (TAB) WIDGET'LARI === */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    border-top: none; /* Üst kenarlık TabBar ile birleşir */
    background: {CONTENT_BACKGROUND};
    padding: 12px;
    border-bottom-left-radius: 4px;
    border-bottom-right-radius: 4px;
}}
QTabBar::tab {{
    background: transparent; /* Pasif sekme arka planı */
    color: {TEXT_SECONDARY};
    border: 1px solid transparent; /* Sadece seçili olana kenarlık */
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 1px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 70px;
    font-weight: 500;
}}
QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
    background-color: {HOVER_BACKGROUND};
}}
QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    background-color: {CONTENT_BACKGROUND}; /* Seçili sekme pane ile aynı renk */
    /* border: 1px solid {BORDER_COLOR}; */ /* Kenarlık pane ile birleşiyor */
    /* border-bottom-color: {CONTENT_BACKGROUND}; /* Alt kenarlığı pane ile birleştir */
    font-weight: 600;
    border-bottom: 2px solid {ACCENT_PRIMARY}; /* Veya alttan vurgu */
}}
QTabBar::tab:disabled {{ color: {TEXT_SECONDARY}; background-color: transparent; }}
QTabWidget::tab-bar {{ alignment: left; /* left: 5px; */ }}

/* === GRUP KUTUSU (GROUPBOX) === */
QGroupBox {{
    background-color: transparent;
    border: 1px solid {BORDER_COLOR};
    border-radius: 5px;
    margin-top: 22px; /* Başlık için üstte boşluk */
    padding: 15px 10px 10px 10px;
    font-weight: 500;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px; /* Başlık içi boşluk */
    margin-left: 10px;
    background-color: {ACCENT_PRIMARY};
    color: {TEXT_ON_ACCENT};
    border-radius: 3px;
    font-weight: 600;
}}

/* === MENÜ ÇUBUĞU VE MENÜLER === */
QMenuBar {{
    background-color: {APP_BACKGROUND};
    color: {TEXT_PRIMARY};
    padding: 3px;
    border-bottom: 1px solid {BORDER_COLOR};
    font-weight: 500;
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{ background-color: {ACCENT_PRIMARY}; color: {TEXT_ON_ACCENT}; }}
QMenu {{
    background-color: {INPUT_BACKGROUND}; /* Menüler giriş alanları gibi daha koyu */
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 5px;
    font-weight: 500;
}}
QMenu::item {{
    padding: 7px 22px 7px 15px;
    border-radius: 3px;
    min-width: 160px;
}}
QMenu::item:selected {{ background-color: {ACCENT_PRIMARY}; color: {TEXT_ON_ACCENT}; }}
QMenu::item:disabled {{ color: {TEXT_SECONDARY}; background-color: transparent; }}
QMenu::separator {{ height: 1px; background: {BORDER_COLOR}; margin: 5px 0px; }}
QMenu::icon {{ padding-left: 5px; /* Varsa ikon için boşluk */ }}

/* === DURUM ÇUBUĞU (STATUS BAR) === */
QStatusBar {{
    background-color: {APP_BACKGROUND};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER_COLOR};
    padding: 4px;
    font-weight: 500;
}}
QStatusBar::item {{ border: none; }}

/* === ARAÇ İPUCU (TOOLTIP) === */
QToolTip {{
    background-color: {APP_BACKGROUND}; /* Ana uygulama arka planı gibi */
    color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT_PRIMARY}; /* Vurgu renginde kenarlık */
    padding: 6px;
    border-radius: 4px;
    opacity: 245;
    font-weight: normal;
}}

/* === SCROLL BARLAR === */
QScrollBar:horizontal {{
    border: none; background: {APP_BACKGROUND}; height: 12px; margin: 0px 0px 0px 0px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_COLOR}; min-width: 25px; border-radius: 6px;
}}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT_PRIMARY}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ border: none; background: none; width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

QScrollBar:vertical {{
    border: none; background: {APP_BACKGROUND}; width: 12px; margin: 0px 0px 0px 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COLOR}; min-height: 25px; border-radius: 6px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_PRIMARY}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ border: none; background: none; height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* === Liste/Ağaç/Tablo Görünümleri (Örnek Stil) === */
QTreeView, QListView, QTableView {{
    background-color: {INPUT_BACKGROUND}; /* Liste arka planı */
    alternate-background-color: {CONTENT_BACKGROUND}; /* Satır araları için farklı renk */
    border: 1px solid {BORDER_INPUT};
    border-radius: 4px;
    font-weight: 500;
}}
QTreeView::item, QListView::item, QTableView::item {{
    padding: 5px;
    border-radius: 3px; /* Her item için (isteğe bağlı) */
}}
QTreeView::item:hover, QListView::item:hover, QTableView::item:hover {{
    background-color: {HOVER_BACKGROUND};
}}
QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
    background-color: {ACCENT_PRIMARY};
    color: {TEXT_ON_ACCENT};
}}
QHeaderView::section {{ /* Tablo başlıkları */
    background-color: {CONTENT_BACKGROUND};
    color: {TEXT_PRIMARY};
    padding: 6px;
    border: 1px solid {BORDER_COLOR};
    border-left: none; /* Sol kenarlığı kaldır */
    font-weight: 600;
}}
QHeaderView::section:first {{ border-left: 1px solid {BORDER_COLOR}; }} /* İlk başlığa sol kenarlık */

"""