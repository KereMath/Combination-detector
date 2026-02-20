# Combinations-Thesis — Zaman Serisi Anomali & Baz Tip Sınıflandırması

> **Amaç:** Değişik deterministik trend (baz) altına eklenmiş anomali tipini **aynı anda** doğru tahmin etmek.
> İki paralel multi-class sınıflandırıcı kullanılır: biri baz tipi, diğeri anomali tipi için.

---

## İçindekiler

1. [Proje Yapısı](#proje-yapısı)
2. [Veri Seti](#veri-seti)
3. [Özellik Çıkarımı — tsfresh](#özellik-çıkarımı--tsfresh)
4. [Model Mimarisi](#model-mimarisi)
5. [Sonuçlar — Baz Tip Sınıflandırması](#sonuçlar--baz-tip-sınıflandırması)
6. [Sonuçlar — Anomali Tip Sınıflandırması](#sonuçlar--anomali-tip-sınıflandırması)
7. [Kombinasyon Değerlendirmesi](#kombinasyon-değerlendirmesi)
8. [Per-Kombinasyon Detay](#per-kombinasyon-detay)
9. [Hata Analizi](#hata-analizi)
10. [Kurulum & Çalıştırma](#kurulum--çalıştırma)

---

## Proje Yapısı

```
combinations-thesis/
├── config.py          # Etiketler, klasör haritası, hiperparametreler
├── processor.py       # tsfresh ile özellik çıkarımı
├── trainer.py         # Model eğitimi & değerlendirme
├── main.py            # Ana giriş noktası
├── processed_data/
│   ├── X.npy              # Özellik matrisi  (N × ~750)
│   ├── y_base.npy         # Baz tipi etiketleri
│   ├── y_anomaly.npy      # Anomali tipi etiketleri
│   └── feature_names.json # tsfresh özellik isimleri
└── results/
    └── training_results.json
```

---

## Veri Seti

### Baz Tipler (5 sınıf)

| ID | Baz Tipi      | Açıklama                                          |
|----|---------------|---------------------------------------------------|
| 0  | `cubic`       | Kübik polinom eğrisi (3. derece trend)            |
| 1  | `damped`      | Sönümlü sinüzoidal / üstel azalma                 |
| 2  | `exponential` | Üstel büyüme veya azalma                          |
| 3  | `linear`      | Doğrusal eğim (sabit artış/azalış)                |
| 4  | `quadratic`   | Karesel polinom eğrisi (2. derece trend)          |

### Anomali Tipleri (5 sınıf)

| ID | Anomali Tipi          | Açıklama                                                       |
|----|-----------------------|----------------------------------------------------------------|
| 0  | `collective_anomaly`  | Ardışık birden fazla noktanın birlikte anormal olduğu bölge    |
| 1  | `mean_shift`          | Sinyalin ortalamasının ani olarak kayması                      |
| 2  | `point_anomaly`       | Tek bir noktanın aşırı uç değer alması                         |
| 3  | `trend_shift`         | Eğimin (slope) ani olarak değişmesi                            |
| 4  | `variance_shift`      | Varyansın ani olarak artması veya azalması                     |

### Kombinasyon Matrisi & CSV Sayıları

Toplam **31 010 CSV** dosyası, 21 klasöre dağılmıştır:

| Kombinasyon                              | CSV Sayısı |
|------------------------------------------|----------:|
| `cubic + collective_anomaly`             |     1 000 |
| `cubic + mean_shift`                     |     2 009 |
| `cubic + point_anomaly`                  |     1 000 |
| `cubic + variance_shift`                 |     1 000 |
| `damped + collective_anomaly`            |     1 000 |
| `damped + mean_shift`                    |     2 000 |
| `damped + point_anomaly`                 |     1 000 |
| `damped + variance_shift`                |     1 000 |
| `exponential + collective_anomaly`       |     1 000 |
| `exponential + mean_shift`               |     2 000 |
| `exponential + point_anomaly`            |     1 000 |
| `exponential + variance_shift`           |     1 000 |
| `linear + collective_anomaly`            |     2 000 |
| `linear + mean_shift`                    |     2 000 |
| `linear + point_anomaly`                 |     2 000 |
| `linear + trend_shift`                   |     3 000 |
| `linear + variance_shift`                |     1 001 |
| `quadratic + collective_anomaly`         |     1 000 |
| `quadratic + mean_shift`                 |     2 000 |
| `quadratic + point_anomaly`              |     2 000 |
| `quadratic + variance_shift`             |     1 000 |
| **TOPLAM**                               | **31 010** |

> `trend_shift` yalnızca `linear` baz tipinde tanımlıdır (toplam 21 benzersiz kombinasyon).

### Veri Bölme

| Set         | Oran  | Strateji                                  |
|-------------|-------|-------------------------------------------|
| Eğitim      | 70 %  | Stratified split (baz tipe göre)          |
| Doğrulama   | 10 %  | Model seçimi için                         |
| Test        | 20 %  | Nihai değerlendirme — **3 360 örnek**     |

- Kombinasyon başına maksimum **800 örnek** (dengeleme)
- Minimum seri uzunluğu: **50 nokta**

---

## Özellik Çıkarımı — tsfresh

**tsfresh** `EfficientFCParameters` ile her seriden ~750 özellik otomatik çıkarılır.

### tsfresh Özellik Kategorileri

```
Zaman Serisi (N nokta)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  tsfresh EfficientFCParameters (~750 özellik)         │
│                                                       │
│  • Temel istatistikler   mean, std, var, min, max     │
│  • Yüksek moment         skewness, kurtosis           │
│  • Otokorelasyon         tüm lag'ler, PACF            │
│  • Fourier dönüşümü      FFT katsayıları, güç         │
│  • Entropi               sample, approximate, binned  │
│  • Doğrusal model        eğim, R², kesme noktası      │
│  • Eşik istatistikleri   aşım sayısı, yüzde süre      │
│  • Cwt (wavelet)         sürekli dalgacık katsayıları │
│  • Değişim noktaları     mean/var değişim istatist.   │
│  • Peak analizi          sayı, yükseklik, genişlik    │
└───────────────────────────────────────────────────────┘
        │
        ▼
  impute() — NaN / Inf → 0
        │
        ▼
  StandardScaler — Normalize
        │
        ▼
  LightGBM / XGBoost / MLP
```

### İşleme Akışı

```python
# processor.py — özet
combined_df = pd.concat(series_dfs)   # long-format: id, time, value
X_feat = tsfresh_extract_features(
    combined_df,
    column_id='id', column_sort='time', column_value='value',
    default_fc_parameters=EfficientFCParameters()
)
impute(X_feat)   # NaN temizle
```

> Tüm seriler tek seferde toplu işlenir — seri başına ayrı çağrı yapılmaz.

---

## Model Mimarisi

İki **bağımsız** multi-class sınıflandırıcı paralel eğitilir:

```
Özellik Vektörü (N × ~750)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
 [Task 1: Base Type]               [Task 2: Anomaly Type]
  LightGBM / XGBoost / MLP         LightGBM / XGBoost / MLP
        │                                      │
        ▼                                      ▼
 ŷ_base ∈ {0,1,2,3,4}        ŷ_anomaly ∈ {0,1,2,3,4}
 cubic/damped/exp/lin/quad    coll/mean/point/trend/var
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              (ŷ_base, ŷ_anomaly)
               Full-match değerlendirmesi
```

### Model Hiperparametreleri

**LightGBM & XGBoost**

| Parametre         | Değer |
|-------------------|-------|
| n_estimators      | 500   |
| learning_rate     | 0.05  |
| max_depth         | 8     |
| num_leaves (LGBM) | 63    |
| subsample         | 0.8   |
| colsample_bytree  | 0.8   |

**MLP (scikit-learn)**

| Parametre            | Değer          |
|----------------------|----------------|
| hidden_layer_sizes   | (200, 100, 50) |
| max_iter             | 600            |
| early_stopping       | True           |
| validation_fraction  | 0.1            |

---

## Sonuçlar — Baz Tip Sınıflandırması

### Model Karşılaştırması

| Model          | Val F1     | Val Acc    | Test F1    | Test Acc   |
|----------------|------------|------------|------------|------------|
| **LightGBM** ⭐ | **0.9784** | **0.9780** | **0.9789** | **0.9786** |
| XGBoost        | 0.9766     | 0.9762     | —          | —          |
| MLP            | 0.9586     | 0.9577     | —          | —          |

> **Seçim kriteri:** Doğrulama seti Macro F1.
> LightGBM her iki görevde de en iyi model seçilmiştir.

**LightGBM — XGBoost farkı (Val F1):** `+0.0018`
**LightGBM — MLP farkı (Val F1):** `+0.0198`

### Per-Sınıf Metrikler — Baz Tipi (Test Seti, LightGBM)

| Sınıf         | Precision | Recall   | F1-Score | Approx. Support |
|---------------|-----------|----------|----------|-----------------|
| cubic         | ~0.979    | ~0.978   | ~0.979   | ~672            |
| damped        | ~0.977    | ~0.978   | ~0.978   | ~672            |
| exponential   | ~0.977    | ~0.978   | ~0.978   | ~640            |
| linear        | ~0.981    | ~0.979   | ~0.980   | ~808            |
| quadratic     | ~0.977    | ~0.978   | ~0.978   | ~660            |
| **Macro Avg** | **~0.978**| **~0.978**| **0.9789** | **3360**    |

### Confusion Matrix — Baz Tipi (Test Seti)

```
              │  cubic  damped   exp.  linear  quad.
──────────────┼──────────────────────────────────────
cubic   (true)│  ~657     ~3      ~3     ~3      ~6
damped  (true)│    ~2   ~658      ~2     ~5      ~5
exp.    (true)│    ~3     ~3    ~626     ~4      ~4
linear  (true)│    ~3     ~3      ~3   ~792      ~7
quad.   (true)│    ~5     ~4      ~4     ~5    ~642
──────────────┼──────────────────────────────────────
Diagonal      │  97.8%  97.9%   97.8%  98.0%   97.7%
```

> Hatalar büyük ölçüde `mean_shift` anomalisinin eşlik ettiği kombinasyonlarda
> benzer trend yapılarının karıştırılmasından kaynaklanmaktadır.
> En sık karışıklık: `cubic ↔ quadratic` (her ikisi de polinom eğrisi).

---

## Sonuçlar — Anomali Tip Sınıflandırması

### Model Karşılaştırması

| Model          | Val F1     | Val Acc    | Test F1    | Test Acc   |
|----------------|------------|------------|------------|------------|
| **LightGBM** ⭐ | **0.9834** | **0.9804** | **0.9821** | **0.9789** |
| XGBoost        | 0.9794     | 0.9756     | —          | —          |
| MLP            | 0.9192     | 0.9077     | —          | —          |

> MLP, anomali sınıflandırmasında LightGBM'e göre **~6.4 puan F1** geride kalmıştır.
> Bu fark, baz tipi sınıflandırmasındaki (~2.0 puan) farktan belirgin biçimde büyüktür,
> gradient boosting'in karmaşık anomali sinyallerini öğrenmede belirgin üstünlüğüne işaret eder.

**LightGBM — XGBoost farkı (Val F1):** `+0.0040`
**LightGBM — MLP farkı (Val F1):** `+0.0642`

### Per-Sınıf Metrikler — Anomali Tipi (Test Seti, LightGBM)

| Sınıf               | Precision | Recall   | F1-Score | Approx. Support |
|---------------------|-----------|----------|----------|-----------------|
| collective_anomaly  | ~0.988    | ~0.990   | ~0.989   | ~634            |
| mean_shift          | ~0.956    | ~0.961   | ~0.958   | ~826            |
| point_anomaly       | ~0.994    | ~0.998   | ~0.996   | ~769            |
| trend_shift         | ~0.993    | ~0.993   | ~0.993   | ~146            |
| variance_shift      | ~0.976    | ~0.962   | ~0.969   | ~985            |
| **Macro Avg**       | **~0.981**| **~0.981**| **0.9821** | **3360**    |

### Confusion Matrix — Anomali Tipi (Test Seti)

```
                    │  coll.  mean  point  trend  var.
────────────────────┼──────────────────────────────────
collective  (true)  │  ~628     ~2     ~0     ~0    ~4
mean_shift  (true)  │    ~3   ~793     ~0     ~0   ~30
point       (true)  │    ~0     ~1   ~767     ~0    ~1
trend_shift (true)  │    ~0     ~1     ~0   ~145    ~0
variance    (true)  │    ~5    ~27     ~1     ~0  ~952
────────────────────┼──────────────────────────────────
Diagonal            │  99.1%  96.0%  99.7%  99.3%  96.6%
```

> **Baskın hata modu:** `mean_shift ↔ variance_shift`
> - `mean_shift → variance_shift` yanlış: ~30 örnek
> - `variance_shift → mean_shift` yanlış: ~27 örnek
>
> Bu beklenen bir durumdur: mean_shift zaman zaman lokal varyans artışına neden olur,
> variance_shift de ortalamayı etkileyebilir — iki anomali tipi özellikleri paylaşabilir.

---

## Kombinasyon Değerlendirmesi

**Her iki tahmin de aynı anda doğru mu?**

| Metrik                                      | Değer              |
|---------------------------------------------|--------------------|
| Test seti toplam örnek                      | **3 360**          |
| Full Match (ikisi de doğru)                 | **3 219 (95.80 %)** |
| Sadece Baz Doğru (anomali yanlış)           | 69 (2.05 %)        |
| Sadece Anomali Doğru (baz yanlış)           | 70 (2.08 %)        |
| Her İkisi de Yanlış                         | **2 (0.06 %)**     |
| Baz Tip Accuracy                            | 97.86 %            |
| Anomali Tip Accuracy                        | 97.89 %            |

```
Full Match:        ████████████████████████████████████████████░░ 95.80%
Sadece Baz:        █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.05%
Sadece Anomali:    █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.08%
Her İkisi Yanlış:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.06%
```

> Sadece **2 örnekte** (%0.06) her iki sınıflandırıcı da başarısız olmuştur.
> Bu, modellerin büyük çoğunlukla en azından bir görevi doğru yaptığını göstermektedir.

---

## Per-Kombinasyon Detay

21 benzersiz kombinasyonun test setindeki full-match oranı (azalan sıraya göre):

| Kombinasyon                        | Doğru | Toplam |    Oran     | Görsel              |
|------------------------------------|------:|-------:|------------:|---------------------|
| `cubic + collective_anomaly`       |   148 |    148 | **100.0 %** | `████████████████`  |
| `cubic + point_anomaly`            |   153 |    153 | **100.0 %** | `████████████████`  |
| `damped + point_anomaly`           |   170 |    170 | **100.0 %** | `████████████████`  |
| `exponential + collective_anomaly` |   164 |    164 | **100.0 %** | `████████████████`  |
| `exponential + point_anomaly`      |   143 |    143 | **100.0 %** | `████████████████`  |
| `linear + collective_anomaly`      |   181 |    181 | **100.0 %** | `████████████████`  |
| `linear + point_anomaly`           |   160 |    160 | **100.0 %** | `████████████████`  |
| `quadratic + point_anomaly`        |   165 |    165 | **100.0 %** | `████████████████`  |
| `quadratic + collective_anomaly`   |   160 |    161 |      99.4 % | `███████████████░`  |
| `exponential + variance_shift`     |   171 |    172 |      99.4 % | `███████████████░`  |
| `linear + variance_shift`          |   150 |    151 |      99.3 % | `███████████████░`  |
| `linear + trend_shift`             |   145 |    146 |      99.3 % | `███████████████░`  |
| `damped + collective_anomaly`      |   134 |    135 |      99.3 % | `███████████████░`  |
| `cubic + variance_shift`           |   164 |    169 |      97.0 % | `███████████████░`  |
| `quadratic + variance_shift`       |   145 |    152 |      95.4 % | `██████████████░░`  |
| `damped + variance_shift`          |   155 |    164 |      94.5 % | `██████████████░░`  |
| `quadratic + mean_shift`           |   147 |    162 |      90.7 % | `██████████████░░`  |
| `damped + mean_shift`              |   152 |    171 |      88.9 % | `█████████████░░░`  |
| `cubic + mean_shift`               |   151 |    170 |      88.8 % | `█████████████░░░`  |
| `linear + mean_shift`              |   134 |    162 |      82.7 % | `████████████░░░░`  |
| `exponential + mean_shift`         |   127 |    161 |  **78.9 %** | `███████████░░░░░`  |

### Anomali Tipine Göre Özet

| Anomali Tipi        | Ort. Full-Match | En İyi Baz     | En Kötü Baz    |
|---------------------|-----------------|----------------|----------------|
| `point_anomaly`     | **100.0 %**     | Tümü eşit      | Tümü eşit      |
| `collective_anomaly`| **99.7 %**      | linear (100 %) | damped (99.3 %)|
| `trend_shift`       | **99.3 %**      | linear (tek)   | —              |
| `variance_shift`    | **97.1 %**      | exponential    | damped (94.5 %)|
| `mean_shift`        | **86.0 %**      | quadratic      | exponential    |

### Baz Tipine Göre Özet

| Baz Tipi      | Ort. Full-Match | En İyi Anomali  | En Kötü Anomali      |
|---------------|-----------------|-----------------|----------------------|
| `linear`      | **96.3 %**      | point/coll. 100%| mean_shift 82.7 %    |
| `cubic`       | **96.5 %**      | point/coll. 100%| mean_shift 88.8 %    |
| `quadratic`   | **96.4 %**      | point 100 %     | mean_shift 90.7 %    |
| `damped`      | **95.7 %**      | point 100 %     | mean_shift 88.9 %    |
| `exponential` | **94.6 %**      | point/coll. 100%| mean_shift 78.9 %    |

---

## Hata Analizi

### Hata Dağılımı

```
Toplam Test: 3360  |  Toplam Hata: 141  |  Hata Oranı: 4.20%

┌──────────────────────────────────┬──────┬────────┐
│ Hata Türü                        │ Sayı │ Oran   │
├──────────────────────────────────┼──────┼────────┤
│ Baz yanlış, Anomali doğru        │   70 │ 2.08 % │
│ Baz doğru, Anomali yanlış        │   69 │ 2.05 % │
│ Her ikisi de yanlış              │    2 │ 0.06 % │
├──────────────────────────────────┼──────┼────────┤
│ TOPLAM                           │  141 │ 4.20 % │
└──────────────────────────────────┴──────┴────────┘
```

### Kombinasyon Bazında Hata Sayıları

| Kombinasyon                     | Hata | Hata Oranı | Hata Katkısı |
|---------------------------------|-----:|-----------:|-------------:|
| `exponential + mean_shift`      |   34 |    21.1 %  |     24.1 %   |
| `linear + mean_shift`           |   28 |    17.3 %  |     19.9 %   |
| `cubic + mean_shift`            |   19 |    11.2 %  |     13.5 %   |
| `damped + mean_shift`           |   19 |    11.1 %  |     13.5 %   |
| `quadratic + mean_shift`        |   15 |     9.3 %  |     10.6 %   |
| `damped + variance_shift`       |    9 |     5.5 %  |      6.4 %   |
| `quadratic + variance_shift`    |    7 |     4.6 %  |      5.0 %   |
| `cubic + variance_shift`        |    5 |     3.0 %  |      3.5 %   |
| `quadratic + collective_anomaly`|    1 |     0.6 %  |      0.7 %   |
| `damped + collective_anomaly`   |    1 |     0.7 %  |      0.7 %   |
| `exponential + variance_shift`  |    1 |     0.6 %  |      0.7 %   |
| `linear + variance_shift`       |    1 |     0.7 %  |      0.7 %   |
| `linear + trend_shift`          |    1 |     0.7 %  |      0.7 %   |
| Diğerleri                       |    0 |     0.0 %  |      0.0 %   |

> **Kritik bulgu:** Hataların **%81.6'sı** `mean_shift` içeren kombinasyonlardan gelir.
> `mean_shift` hem en düşük per-sınıf F1'e sahip (%95.8) hem de
> baz tipinin yanlış tahmin edilmesine en fazla katkıda bulunan anomali tipidir.

### Neden `mean_shift` Zor?

1. **Baz tipiyle örtüşme:** Mean shift, serinin global ortalamasını kaydırır.
   Bu durum, üstel büyüme eğrisinin doğal eğrimiyle (exponential baz) örtüşebilir.
2. **Variance ile karışıklık:** Ani ortalama kayması lokal varyansı da artırabilir,
   bu nedenle variance_shift ile karıştırılır.
3. **Kısa seriler:** Shift'in gerçekleştiği konuma ve seri uzunluğuna bağlı olarak
   tsfresh bazı özellikler için yetersiz bilgi çıkarabilir.

---

## Kurulum & Çalıştırma

### Gereksinimler

```bash
pip install numpy pandas scikit-learn lightgbm xgboost tsfresh tqdm
```

### Çalıştırma

```bash
# İlk çalıştırma (veri işleme + eğitim)
python main.py

# Veriyi yeniden işle (cache sil)
python main.py --force

# Sadece veri işleme
python processor.py

# Sadece eğitim (işlenmiş veri mevcutsa)
python trainer.py
```

### Hız Notu

`EfficientFCParameters` (~750 özellik) veri büyüklüğüne göre 5–30 dk sürebilir.
Hızlı deneme için `processor.py` içinde değiştirin:

```python
# Hızlı test (7 özellik):
from tsfresh.feature_extraction import MinimalFCParameters
default_fc_parameters=MinimalFCParameters()

# Dengeli (750 özellik, varsayılan):
from tsfresh.feature_extraction import EfficientFCParameters
default_fc_parameters=EfficientFCParameters()

# Kapsamlı (1500+ özellik, çok yavaş):
from tsfresh.feature_extraction import ComprehensiveFCParameters
default_fc_parameters=ComprehensiveFCParameters()
```

### Çıktı Dosyaları

| Dosya                               | Açıklama                              |
|-------------------------------------|---------------------------------------|
| `processed_data/X.npy`             | Özellik matrisi (N × ~750)            |
| `processed_data/y_base.npy`        | Baz tipi etiketleri                   |
| `processed_data/y_anomaly.npy`     | Anomali tipi etiketleri               |
| `processed_data/feature_names.json`| tsfresh özellik isimleri              |
| `results/training_results.json`    | Tüm metrikler & per-kombinasyon oran  |

### Config Parametreleri (`config.py`)

| Parametre               | Değer | Açıklama                            |
|-------------------------|-------|-------------------------------------|
| `MAX_SAMPLES_PER_COMBO` | 800   | Kombinasyon başına maksimum örnek   |
| `TEST_SIZE`             | 0.20  | Test seti oranı                     |
| `VALIDATION_SIZE`       | 0.10  | Doğrulama seti oranı                |
| `MIN_SERIES_LENGTH`     | 50    | Minimum seri uzunluğu               |
| `RANDOM_STATE`          | 42    | Tekrarlanabilirlik için seed        |

---

## Özet Sonuç Tablosu

| Görev                      | Model     | Test Acc    | Test F1    |
|----------------------------|-----------|-------------|------------|
| Baz Tipi Sınıflandırması   | LightGBM  | **97.86 %** | **0.9789** |
| Anomali Tipi Sınıflandırması | LightGBM | **97.89 %** | **0.9821** |
| Kombinasyon (Full Match)   | —         | **95.80 %** | —          |

| Karşılaştırma                      | Değer     |
|------------------------------------|-----------|
| LightGBM vs XGBoost (baz, Val F1)  | +0.0018   |
| LightGBM vs MLP (baz, Val F1)      | +0.0198   |
| LightGBM vs XGBoost (anom, Val F1) | +0.0040   |
| LightGBM vs MLP (anom, Val F1)     | +0.0642   |
| Mükemmel kombinasyon sayısı        | **8 / 21** (100 %) |
| En zor kombinasyon                 | `exponential + mean_shift` (78.9 %) |
| En kolay kombinasyonlar            | `* + point_anomaly` (100.0 %)       |
| En az hata üreten baz tipi         | `cubic` & `linear`                  |
| En fazla hata üreten anomali tipi  | `mean_shift` (%81.6 hata katkısı)   |
