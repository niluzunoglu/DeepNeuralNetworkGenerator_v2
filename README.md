# NeuralNetworkGenerator (v2)

[English](#EN) | [Türkçe](#TR)

## TR

Neural Network Generator, bir arayüz kullanarak yapay sinir ağlarının oluşturulmasını ve görselleştirebilmesini sağlayan bir çalışmadır.

Bu uygulama sayesinde, kullanıcılar katman sayısından katman türlerine (giriş, çıkış, gizli katman gibi), aktivasyon fonksiyonlarından giriş/çıkış katmanlarındaki nöron sayısına kadar pek çok detayı bir arayüz kullanarak seçebilir ve kendi tasarladıkları nöral ağ ile çalışmalar yapabilirler.

Arka planda ise kullanıcı tercihlerini temel alarak sinir ağını kuran bir altyapı çalışır. Bu altyapıda hazır derin öğrenme fonksiyonları kullanılmayacak, nöral ağ ve içerisinde kullanılan fonksiyonlar (aktivasyon, kayıp fonksiyonları gibi) temelden implemente edilecektir.

Uygulamanın frontend bölümü PyQt6 kullanılarak geliştirilmiştir. Backend bölümüne implementasyonlar numpy ve pandas kullanarak yapılmıştır.

*Projenize `screenshots` adında bir klasör oluşturup, `app_interface.png` gibi bir isimle ekran görüntünüzü oraya kaydedin ve yukarıdaki satırdaki yorumu kaldırarak aktif hale getirin.*

### Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/niluzunoglu/NeuralNetworkGenerator.git
    cd NeuralNetworkGenerator
    ```

2.  **Sanal Ortam Oluşturun ve Aktive Edin (Önerilir):**
    ```bash
    python -m venv venv

    # Windows için aktivasyon:
    # venv\Scripts\activate
    # macOS/Linux için aktivasyon:
    # source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje ana dizininde bir `requirements.txt` dosyası bulunmaktadır. Aşağıdaki komutla gerekli kütüphaneleri bu dosyayı kullanarak yükleyebilirsiniz.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı Çalıştırın:**
    Projenin ana dizinindeyken (`app.py` dosyasının bulunduğu dizin):
    ```bash
    python app.py
    ```

### Nasıl Kullanılır?

1.  Uygulamayı başlattığınızda, "Genel Parametreler" bölümünden öğrenme oranı, epoch sayısı, kayıp fonksiyonu ve toplam katman sayısını ayarlayın.
2.  "Toplam Katman Sayısı" değiştikçe, "Katman Detayları" bölümünde her katman için nöron sayısı ve aktivasyon fonksiyonunu seçin.
3.  Ayarlarınızı tamamladıktan sonra, "İşlemler" bölümündeki:
    *   **"Modeli Oluştur ve Eğit"** butonuna tıklayarak girdiğiniz parametrelerle bir sinir ağı oluşturulmasını (ve ileride eğitilmesini) sağlayabilirsiniz. Oluşturulan parametreler "Çıktı ve Loglar" alanında gösterilecektir.
    *   **"Model Parametrelerini Sıfırla"** butonuna tıklayarak tüm ayarları varsayılan değerlerine döndürebilirsiniz.
4.  "Çıktı ve Loglar" bölümünden ağ oluşturma ve eğitim süreciyle ilgili bilgileri takip edebilirsiniz.


Sorularınız için nil.uzunoglu@std.yildiz.edu.tr adresinden ulaşabilirsiniz.

## EN

Neural Network Generator is a project that enables the creation and visualization of artificial neural networks through a graphical user interface.

With this application, users can customize many aspects of their neural networks, including the number of layers, layer types (such as input, output, and hidden layers), activation functions, and the number of neurons in the input and output layers — all through an interactive UI. Users can then work with the neural network they have designed.

In the background, a backend infrastructure dynamically constructs the neural network based on user selections. Instead of using prebuilt deep learning frameworks, the neural network and all internal functions (such as activation and loss functions) are implemented from scratch.

The frontend of the application is built using PyQt6, while the backend implementation is developed with NumPy and Pandas.

For any questions, feel free to contact: nil.uzunoglu@std.yildiz.edu.tr