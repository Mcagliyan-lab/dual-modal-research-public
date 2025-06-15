# 🧮 Modül Bazında Formül Detayları

## NN-EEG Modülü
### Temporal Sinyal Çıkarımı
```math
s_t^{(l)} = \frac{1}{N^{(l)}} \sum_{i=1}^{N^{(l)}} |a_{i,t}^{(l)}|
```
**Kod Implementasyonu:**
```python
# src/nn_eeg/signal_extraction.py
def extract_temporal_signal(layer_activations):
    return np.mean(np.abs(layer_activations), axis=1)
```

**Parametreler:**
- `layer_activations`: (batch_size, num_neurons) boyutlu aktivasyon matrisi
- **Return**: (batch_size,) boyutlu temporal sinyal

## NN-fMRI Modülü
### ζ-skor Hesaplama
```math
\zeta(g) = \mathbb{E}[f(S \cup \{g\}) - f(S)]
```
**Kod Implementasyonu:**
```python
# src/nn_fmri/impact_analysis.py
def calculate_zeta(grid, model, n_samples=1000):
    base_score = model.evaluate()
    total_impact = 0
    for _ in range(n_samples):
        masked = mask_random_grids(model, exclude=grid)
        total_impact += base_score - masked.score
    return total_impact / n_samples
```

**Optimizasyon:**
- Paralel hesaplama desteği
- Önbellekleme mekanizması 