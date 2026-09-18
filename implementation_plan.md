# Kestirimci Bakım Sistemini LSTM ve Zaman Serisi Analizi ile Derinleştirme

Bu plan, mevcut Kestirimci Bakım Sistemine bir zaman serisi modeli (LSTM) ve gerçek bir endüstriyel veri seti (NASA Turbofan/CMAPSS) eklemeye yönelik adımları ana hatlarıyla belirtmektedir. Amaç, mevcut anlık veri tabanlı yaklaşım (AI4I üzerinde Random Forest) ile sıralı (zaman serisi) bir yaklaşım (CMAPSS üzerinde LSTM) arasında karşılaştırmalı bir analiz yapmak ve aralarındaki ödünleşimler (trade-offs) üzerine profesyonel bir sonuç bölümü yazmaktır.

## Kullanıcı İncelemesi Gerekli

> [!IMPORTANT]
> CMAPSS veri seti tipik olarak Kaggle'dan veya NASA'nın kendi veri deposundan indirilir. Bunu nasıl çekeceğimize karar vermemiz gerekiyor. Ayrıca, LSTM modelini eklemek için Derin Öğrenme (Deep Learning) kütüphanelerine (TensorFlow veya PyTorch) ihtiyacımız olacak.

## Açık Sorular

> [!WARNING]
> Lütfen devam etmeden önce aşağıdaki sorulara geri bildirimde bulunun:
> 1. **Framework:** LSTM modeli için **TensorFlow/Keras** mı yoksa **PyTorch** mu tercih edersiniz?
> 2. **Veri Seti İndirme:** Bu bilgisayarda veri setini programatik olarak indirmek için Kaggle API kimlik bilgileriniz kurulu mu, yoksa manuel olarak indirdiğiniz veri setini `data/` klasörüne koymayı ve benim o şekilde ilerlememi mi tercih edersiniz?
> 3. **Format:** Bu bir "Veri Bilimi (Data Science) Evresi" olduğu için, karşılaştırmalı analizi ve model eğitimini bir **Jupyter Notebook** (`notebooks/LSTM_Comparative_Analysis.ipynb`) içerisinde yapmamız uygun mu? Bu, sonuç bölümünü yazmayı ve grafikleri çizmeyi çok daha kolaylaştıracaktır.

## Önerilen Değişiklikler

### 1. Bağımlılık (Dependency) Güncellemeleri
Derin Öğrenme ve Notebook araçlarını desteklemek için ortamı güncelleyeceğiz.

#### [MODIFY] [requirements.txt](file:///c:/Users/enesu/Desktop/PYTHON/Predictive_Maintenance_System/requirements.txt)
- Jupyter, Matplotlib/Seaborn (yoksa) ve seçilen Derin Öğrenme kütüphanesi (TensorFlow veya PyTorch) eklenecektir.

---

### 2. Veri Seti Edinimi
CMAPSS veri setini işlemek için komut dosyaları veya talimatlar.

#### [NEW] [download_cmapss.py](file:///c:/Users/enesu/Desktop/PYTHON/Predictive_Maintenance_System/src/download_cmapss.py) (Kullanıcı tercihine bağlı olarak opsiyonel)
- CMAPSS veri setini indirip `data/cmapss/` dizinine çıkaracak bir Python betiği.

---

### 3. Veri Bilimi & Karşılaştırmalı Analiz
Veri bilimi görevlerini yerine getirmek için kapsamlı bir notebook.

#### [NEW] [LSTM_Comparative_Analysis.ipynb](file:///c:/Users/enesu/Desktop/PYTHON/Predictive_Maintenance_System/notebooks/LSTM_Comparative_Analysis.ipynb)
- **Veri Ön İşleme (Data Preprocessing):** CMAPSS veri setini yükleme, sensör okumalarını normalize etme ve zaman serisi pencereleri (sliding window/lookback sequences) oluşturma.
- **Temel Model (Random Forest):** Karşılaştırma yapabilmek için aynı veri seti (CMAPSS) üzerinde basit bir Random Forest (Rastgele Orman) modeli eğitme.
- **LSTM Modeli:** Kalan Faydalı Ömür (RUL - Remaining Useful Life) tahmini için bir LSTM ağı kurma ve eğitme.
- **Değerlendirme:** RMSE (Kök Ortalama Kare Hatası), MAE (Ortalama Mutlak Hata) ve hesaplama maliyetlerini karşılaştırma.
- **Sonuç:** "Neden LSTM daha iyi/kötü?" sorusunu detaylıca yanıtlayan bir bölüm. LSTM'in zaman içindeki bozulmaları (temporal degradation) nasıl yakalayabildiğini, buna karşın Random Forest'ın anlık durumlara (point-in-time) bağlı kısıtlamalarını tartışacağız. İşe alım uzmanlarının dikkat edeceği kısım burası olacak.

---

### 4. Dokümantasyon
Yeni veri bilimi evresini yansıtacak şekilde proje dokümantasyonunu güncelleme.

#### [MODIFY] [README.md](file:///c:/Users/enesu/Desktop/PYTHON/Predictive_Maintenance_System/README.md)
- Zaman Serisi / Derin Öğrenme karşılaştırmalı analiz aşamasını detaylandıran ve oluşturduğumuz notebook'a bağlantı veren yeni bir bölüm ekleme.

## Doğrulama Planı

### Otomatik Testler
- Notebook için otomatik birim (unit) testleri planlanmamıştır, ancak Jupyter notebook içindeki tüm hücreler sırayla çalıştırılarak hatasız çalıştıklarından emin olunacaktır.

### Manuel Doğrulama
- Modelin doğru bir şekilde öğrendiğinden emin olmak için notebook'ta oluşturulan grafikler (Kayıp eğrileri, RUL tahmin grafikleri) incelenecektir.
- Sonuç bölümünün iş görüşmelerine uygun, güçlü ve açıklayıcı bir karşılaştırma sunduğundan emin olmak için kontrol edilecektir.
