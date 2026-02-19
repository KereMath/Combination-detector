# Çalışma Notları — Tez Araştırma Günlüğü

---

## Not 1 — Binary → 9-Sınıf Pipeline Neden Patladı?

**Sorun:** Tek tek %90+ F1 elde eden binary detector'lar, birleştirip "en yüksek confidence'a sahip sınıfı seç" diye kullanılınca %30-40'a düştü.

**Teşhis doğru mu?**

Evet, teşhis çok büyük ihtimalle doğru. Aşağıda neden olduğunu detaylı açıklıyorum:

---

### Problem 1: Binary Classifier "Kendi Sınıfı" vs "Her Şey" Öğreniyor

Her detector şu şekilde eğitildi:

```
Mean Shift Detector  → mean_shift (1) vs everything_else (0)
Variance Shift Detector → variance_shift (1) vs everything_else (0)
...
```

"Everything else" içinde diğer anomali tipleri de var. Yani:

- **Mean shift detector**, mean_shift ile variance_shift'i ayırt etmek için değil,
  mean_shift ile [variance_shift + collective + point + ...] arası genel farkı öğreniyor.
- **Variance shift detector** da tam tersini yapıyor.

Bu iki detector aslında **birbirinin rakibini** öğrenmemiş, **genel non-class'ı** öğrenmiş.

---

### Problem 2: Paylaşılan İstatistiksel İmzalar (Ortak Özellikler)

Mean shift ve variance shift için bazı istatistiksel özellikler benzer sinyal verir:

| Özellik | Mean Shift | Variance Shift |
|---------|-----------|----------------|
| `half_mean_diff` | **Güçlü sinyal** | Zayıf sinyal |
| `half_std_diff` | Zayıf sinyal | **Güçlü sinyal** |
| `rolling_mean_std` | **Güçlü sinyal** | **Güçlü sinyal** |
| `rolling_std_mean` | Orta sinyal | **Güçlü sinyal** |
| `autocorr_lag1` | Orta | Orta |

`rolling_mean_std` gibi özellikler **her iki anomali tipinde de** yüksek çıkabilir.
Binary classifier bunu "kendi sınıfım var" olarak okuyabilir.

---

### Problem 3: Confidence Çakışması

Pipeline şöyle karar veriyor:
```
argmax([p_mean_shift, p_variance_shift, ..., p_trend_shift])
```

Eğer gerçek anomali **variance_shift** ise:
- variance_shift detector: conf = 0.82 ✓
- mean_shift detector: conf = 0.78 ← neredeyse aynı!

Fark sadece 0.04 → en küçük gürültü yanlış sınıfı seçtirebilir.

Binary eğitimde bu fark önemli değildi çünkü "var mı yok mu" sorusuna cevap veriyordu.
Multi-class için bu fark ölümcül.

---

### Problem 4: Negatif Örneklerin Dağılımı Tutarsız

Her binary classifier farklı negatif setlerle eğitildi:

```
Mean shift detector'un negatif seti:
  collective + contextual + deterministic + point + stochastic + trend + variance + volatility

Variance shift detector'un negatif seti:
  collective + contextual + deterministic + mean + point + stochastic + trend + volatility
```

Yani mean shift detector'ın negatif setinde **variance shift VAR**,
ama variance shift detector'ın negatif setinde **mean shift VAR**.

Bu asimetri, birbirini "tanımamasına" ve yüksek confidence çakışmasına neden oluyor.

---

### Çözüm Yolları

#### Çözüm A: Doğrudan Multi-class Eğitim (En Basit)

Tek bir 9-class classifier eğit, tüm anomali tiplerini aynı anda gör:

```
Input: zaman serisi özellikleri
Output: [collective, contextual, deterministic, mean_shift, point, stochastic, trend_shift, variance, volatility]
```

Bu yaklaşımda model **anomaliler arası farkları** öğrenir, çünkü eğitim sırasında
mean_shift örneği görürken variance_shift ile karşılaştırılmış oluyor.

#### Çözüm B: Calibrated Ensemble

Her binary classifier'ın output'unu kalibre et (Platt scaling / isotonic regression),
sonra normalize et:
```python
probs_calibrated = calibrate(p_raw)
probs_normalized = probs_calibrated / probs_calibrated.sum()
```

#### Çözüm C: Kombinasyon Verisiyle Eğitim (Bu Proje)

Bu `combinations-thesis` projesi tam da bunu yapıyor:
- Combinations klasöründeki verilerle hem base tip hem anomali tipi aynı anda öğreniliyor
- Model, "bu base trend üstünde mean_shift mi variance_shift mi?" sorusunu
  **aynı training loop içinde** görüyor

---

### Özetle

```
Binary classifier × N + argmax  ≠  Multi-class classifier

Sebep: Her binary model sadece "kendi vs geri kalan" öğreniyor,
       "kendi vs rakipler" değil.
```

---

## Not 2 — İleri Araştırma Fikirleri

### Visual / ECG-style Modeller

- **Time series görüntü dönüşümü**: Gramian Angular Field (GAF), Recurrence Plot, MTF
- Zaman serisini 2D görüntüye çevirip ResNet/EfficientNet ile sınıflandırma
- ECG modellerinden ilham: PhysioNet / MIT-BIH dataset'lerindeki yaklaşımlar
- **Beehive monitoring**: Ses/titreşim anomali tespiti için kullanılan CNN mimarileri

### Transformer / Embedding Yaklaşımlar

- **Time2Vec**: Periyodik ve lineer zaman gömmeleri
- **PatchTST**: Zaman serisini patch'lere böl, her patch = token → Transformer
- **TimesNet**: 2D dönüşüm + Vision Transformer hibrit
- **Moirai (Salesforce)** / **TimesFM (Google)**: Foundation model, fine-tuning ile anomali tespiti
- **Chronos (Amazon)**: Pre-trained time series language model

### Potansiyel Deney Tasarımı

```
1. Mevcut custom features (38 özellik) → LightGBM  [baseline, yapıldı]
2. TSFresh (700+ özellik) → LightGBM              [denenmiş]
3. Raw series → PatchTST (Transformer)             [denenmedi]
4. GAF görüntüsü → ResNet-18                       [denenmedi]
5. Chronos embedding → XGBoost head               [denenmedi]
```

---

## Not 3 — Repo / Data Durumu

**GitHub repos:**
- `ikilimodeller` → gerekenler klasörü, 11 binary detector sonuçları

**Combinations-thesis bu klasör:**
- Combinations klasöründeki ~31k CSV'yi kullanıyor
- 2 ayrı multi-class classifier: base_type + anomaly_type
- Full-match metriği ile değerlendirme

---

*Son güncelleme: 2026-02-19*
