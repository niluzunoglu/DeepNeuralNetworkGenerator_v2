from PyQt6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QTextEdit, QLabel, QFormLayout, QLineEdit, QMessageBox
import numpy as np 

class WeightBiasDialog(QDialog):
    def __init__(self, katman_no, prev_layer_neurons, current_layer_neurons, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Katman {katman_no} - Weight ve  Bias Girişi")
        self.katman_no = katman_no
        self.prev_layer_neurons = prev_layer_neurons 
        self.current_layer_neurons = current_layer_neurons 

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.info_label = QLabel(
            f"Ağırlık matrisi <b>{prev_layer_neurons}x{current_layer_neurons}</b> boyutunda olmalı.<br>"
            f"Bias vektörü <b>1x{current_layer_neurons}</b> boyutunda olmalı.<br>"
            "Her satırdaki değerleri virgülle ayırarak girin. Her ağırlık satırını yeni bir satıra yazın."
        )
        self.info_label.setWordWrap(True)
        self.layout.addWidget(self.info_label)

        self.weights_input = QTextEdit()
        self.weights_input.setPlaceholderText(f"Örnek {prev_layer_neurons}x{current_layer_neurons} ağırlık matrisi:\n0.1, 0.2\n0.3, 0.4\n...")
        self.weights_input.setMinimumHeight(80)
        form_layout.addRow(f"Ağırlıklar ({prev_layer_neurons}x{current_layer_neurons}):", self.weights_input)

        self.biases_input = QLineEdit() 
        self.biases_input.setPlaceholderText(f"Örnek 1x{current_layer_neurons} bias vektörü: 0.01, 0.05,...")
        form_layout.addRow(f"Biaslar (1x{current_layer_neurons}):", self.biases_input)
        
        self.layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject) 
        self.layout.addWidget(self.button_box)

        self.setMinimumWidth(400)

    def get_values(self):

        weights_str = self.weights_input.toPlainText().strip()
        biases_str = self.biases_input.text().strip()
        
        parsed_weights = None
        parsed_biases = None
        error_message = ""

        try:
            if weights_str:
                rows = weights_str.split('\n')
                parsed_weights_list = []
                for row_str in rows:
                    if row_str.strip(): # Boş satırları atla
                        parsed_weights_list.append(list(map(float, row_str.split(','))))
                
                if parsed_weights_list:
                    parsed_weights = np.array(parsed_weights_list)
                    if parsed_weights.shape != (self.prev_layer_neurons, self.current_layer_neurons):
                        error_message += f"Ağırlık matrisi boyutu yanlış! Beklenen: {(self.prev_layer_neurons, self.current_layer_neurons)}, Girilen: {parsed_weights.shape}\n"
                        parsed_weights = None 
        except ValueError:
            error_message += "Ağırlıklar sayısal değerler içermeli ve virgülle ayrılmalı.\n"
            parsed_weights = None
        except Exception as e:
            error_message += f"Ağırlıkları işlerken bir hata oluştu: {e}\n"
            parsed_weights = None

        try:
            if biases_str:
                parsed_biases_list = list(map(float, biases_str.split(',')))
                parsed_biases = np.array([parsed_biases_list]) # (1, num_biases) şeklinde
                if parsed_biases.shape != (1, self.current_layer_neurons):
                    error_message += f"Bias vektörü boyutu yanlış! Beklenen: {(1, self.current_layer_neurons)}, Girilen: {parsed_biases.shape}\n"
                    parsed_biases = None
        except ValueError:
            error_message += "Biaslar sayısal değerler içermeli ve virgülle ayrılmalı.\n"
            parsed_biases = None
        except Exception as e:
            error_message += f"Biasları işlerken bir hata oluştu: {e}\n"
            parsed_biases = None
            
        return parsed_weights, parsed_biases, error_message

    def accept(self):
        weights, biases, error = self.get_values()
        if error:
            QMessageBox.warning(self, "Giriş Hatası", error)
        else:
            super().accept() 