# 📝 Teorik Formüller ve Kod Implementasyonu

Bu belge, Dual-Modal Neural Network Neuroimaging Framework projesinde kullanılan temel matematiksel formüller ile bunların kod tabanındaki karşılık gelen implementasyonları arasındaki ilişkiyi detaylandırmaktadır. Bu sayede, teorik kavramların pratikte nasıl uygulandığına dair şeffaf bir anlayış sağlanmaktadır.

| Formül | Koddaki Karşılığı | Açıklama | Parametreler |
|--------|--------------------|----------|--------------|
| `$s_t^{(l)} = (1/N) \sum |a_{i,t}^{(l)}|$` | `src/nn_neuroimaging/nn_eeg/implementation.py` - `extract_temporal_signals` metodu | Katman aktivasyonlarının zaman serisi gösterimi, EEG sinyallerine benzetilir. | `layer`: Analiz edilen sinir ağı katmanı; `N`: Katmandaki nöron sayısı. |
| `$\zeta(g) = E[f(S \cup \{g\}) - f(S)]$` | `src/nn_neuroimaging/nn_fmri/implementation.py` - `compute_zeta_scores` metodu | Bir grid bölgesinin model çıktısına olan etki derecesini Shapley değerlerinden ilham alarak ölçer. | `g`: 3B grid koordinatı veya bölgesi; `f`: Modelin çıktı fonksiyonu; `S`: Diğer grid bölgelerinin bir alt kümesi. |
| `$P(f)=(1/K) \sum |F_k(f)|^2$` | `src/nn_neuroimaging/nn_eeg/implementation.py` - `analyze_frequency_domain` metodu | Welch metodu kullanılarak bir sinyalin frekans alanındaki güç yoğunluğunu hesaplar. | `f`: Frekans; `K`: Ortalama alınan segment sayısı; `F_k(f)`: $k$-inci pencerenin Fourier dönüşümü; `window_size`: Analiz penceresinin boyutu. |

**Not**: Projedeki tüm önemli implementasyonlar `src/` dizininde bulunmaktadır ve ilgili docstringlerde daha fazla detay sağlanmıştır. 