
yardim_metni = """
<h2>Neural Network Generator</h2>
<p>Bu simülator bir neural network oluşturmayı ve bu network üzerinde temel parametre ayarlarını yapmanızı sağlar.</p>
<p>Arayüzde görünen temel parametrelerin anlamları ve kullanım sınırları aşağıdaki gibidir: </p>

<h3>Genel Parametreler Penceresi:</h3>
<ul>
    <li><b>Öğrenme Oranı:</b> Default değeri 0.1'dir. Her bir artırımda 0.01 artar. En fazla 1 olabilir. Minimum değeri 0.001'dir.</li>
    <li><b>Epoch Sayısı:</b> Tüm eğitim veri setinin networkten kaç kez geçirileceğini belirtir. Varsayılan değeri 10'dur. En az 1, en fazla 500 olabilir. </li>
    <li><b>Kayıp Fonksiyonu:</b> Bulunan sonucun gerçek sonuçtan ne kadar farklı olduğunun ölçüsüdür. MSE, RMSE gibi değerler alabilir. (örn: Mean Squared Error).</li>
    <li><b>Toplam Katman Sayısı:</b> Networkteki katmanların toplam sayısını belirtir. (girdi katmanı hariç, çıktı katmanı dahil). Varsayılan değeri 2'dir. En az 1, en fazla 20 olabilir.</li>
</ul>

<h3>Katman Detayları:</h3>
<p>"Toplam Katman Sayısı" değiştirildiğinde, her katman için aşağıdaki ayarlar yapılabilir:</p>
<p> Katman sayısı değiştikçe katman eklemek için yeni pencereler oluşacaktır, katman detayları penceresini aşağı kaydırarak görebilirsiniz.</p>
<ul>
    <li><b>Nöron Sayısı:</b> O katmanda bulunacak nöron sayısı.Minimum 1, maximum 1024 olabilir. Varsayılan değer 10'dur. </li>
    <li><b>Aktivasyon Fonksiyonu:</b> ReLU, Sigmoid, Tanh gibi aktivasyon fonksiyonu seçilebilir.</li>
    <li><b>Ağırlık Belirleme Ekranı:</b> Weight ve bias'ların girilebileceği ekrandır. </li>
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