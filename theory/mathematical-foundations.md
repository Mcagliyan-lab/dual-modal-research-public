# 📐 Mathematical Foundations

## Notation and Definitions

### Network Structure
- $L$: Number of layers
- $\mathbf{A}^{(l)} \in \mathbb{R}^{B \times N^{(l)}}$: Activations at layer $l$
- $\mathbf{W}^{(l)} \in \mathbb{R}^{N^{(l)} \times N^{(l-1)}}$: Weight matrix between layers
- $N^{(l)}$: Number of neurons in layer $l$

### Temporal Analysis (NN-EEG)

#### Signal Extraction
$$s_t^{(l)} = \frac{1}{N^{(l)}} \sum_{i=1}^{N^{(l)}} |a_{i,t}^{(l)}|$$ (Bkz: `src/nn_neuroimaging/nn_eeg/implementation.py` - `extract_temporal_signals` metodu)

#### Time Series Construction  
$$\mathbf{S}^{(l)} = [s_{t-W+1}^{(l)}, s_{t-W+2}^{(l)}, \ldots, s_t^{(l)}]$$ (Bu kavram, `extract_temporal_signals` metodunda zaman serisi verilerinin oluşturulmasına karşılık gelir.)

#### Power Spectral Density
$$P^{(l)}(f) = \frac{1}{K} \sum_{k=1}^K |F_k^{(l)}(f)|^2$$ (Bkz: `src/nn_neuroimaging/nn_eeg/implementation.py` - `analyze_frequency_domain` metodu)

where $F_k^{(l)}(f)$ is the FFT of the $k$-th window.

#### Frequency Band Power
$$\text{BP}_{\text{band}}^{(l)} = \sum_{f \in \text{band}} P^{(l)}(f)$$ (Bkz: `src/nn_neuroimaging/nn_eeg/implementation.py` - `_extract_band_powers` metodu)

#### State Classification Function
$$\text{State}^{(l)} = \arg\max_s \{\text{BP}_s^{(l)} : s \in \{\delta, \theta, \alpha, \beta, \gamma\}\}$$ (Bkz: `src/nn_neuroimaging/nn_eeg/implementation.py` - `classify_operational_states` metodu)

### Spatial Analysis (NN-fMRI)

#### Grid Partitioning
$$\mathcal{G}^{(l)} = \{\mathcal{N}_{i,j,k} : i \in [g_h], j \in [g_w], k \in [g_c]\}$$ (Bkz: `src/nn_neuroimaging/nn_fmri/implementation.py` - `_partition_into_grids` metodu)

#### Spatial Density Function
$$\phi(g_{i,j,k}) = \frac{1}{|\mathcal{N}_{i,j,k}|} \sum_{(h,w,c) \in \mathcal{N}_{i,j,k}} |a_{h,w,c}^{(l)}| + \lambda \log(\sigma^2_{g_{i,j,k}} + \epsilon)$$ (Bkz: `src/nn_neuroimaging/nn_fmri/implementation.py` - `_calculate_activation_density` metodu)

#### Impact Score (ζ-score)
$$\zeta(g) = \mathbb{E}_{S \subseteq \mathcal{G} \setminus \{g\}} [f(S \cup \{g\}) - f(S)]$$ (Bkz: `src/nn_neuroimaging/nn_fmri/implementation.py` - `compute_zeta_scores` metodu)

#### Monte Carlo Approximation
$$\zeta(g) \approx \frac{1}{K} \sum_{k=1}^K [f(S_k \cup \{g\}) - f(S_k)]$$ (Bkz: `src/nn_neuroimaging/nn_fmri/implementation.py` - `compute_zeta_scores` metodu içindeki örnekleme mantığına karşılık gelir.)

#### Connection Strength
$$C_{A \rightarrow B} = \sum_{i \in A} \sum_{j \in B} |W_{ij}| \cdot \text{ReLU}(a_i) \cdot \sigma'(z_j)$$ (Bu formül, kodda doğrudan bir metodla temsil edilmemektedir, kavramsal bir referanstır.)

### Cross-Modal Integration

#### Consistency Score
$$\text{Consistency}(t) = \frac{1}{3}[\rho_{\gamma\zeta}(t) + \text{Agreement}_{\text{state}}(t) + \text{Coherence}_{\text{anom}}(t)]$$ (Bkz: `src/nn_neuroimaging/integration/framework.py` - `_cross_modal_validation_fixed` metodu)

#### Temporal-Spatial Correlation
$$\rho_{\gamma\zeta}(t) = \text{corr}\left(\text{BP}_{\gamma}^{(l)}(t), \max_g \zeta_g^{(l)}(t)\right)$$ (Bkz: `src/nn_neuroimaging/integration/framework.py` - `_cross_modal_validation_fixed` metodu içindeki korelasyon hesaplaması)

#### State Agreement Rate
$$\text{Agreement}_{\text{state}}(t) = \frac{1}{L} \sum_{l=1}^L \mathbb{I}[\text{State}_{\text{EEG}}^{(l)}(t) = \text{State}_{\text{fMRI}}^{(l)}(t)]$$ (Bkz: `src/nn_neuroimaging/integration/framework.py` - `_cross_modal_validation_fixed` metodu içindeki durum eşleşme kontrolü)

### Information Theory

#### Spectral Entropy
$$H_{\text{spectral}}^{(l)} = -\sum_f P^{(l)}(f) \log P^{(l)}(f)$$ (Bu formül, doğrudan bir kod metodunda bulunmamaktadır, kavramsal bir metrik veya gelecekteki bir ekleme olabilir.)

#### Spatial Information Content
$$I_{\text{spatial}}(g) = H(Y) - H(Y|\phi(g))$$ (Bu formül, doğrudan bir kod metodunda bulunmamaktadır, kavramsal bir metrik veya gelecekteki bir ekleme olabilir.)

#### Mutual Information Between Layers
$$I(L_i; L_j) = \sum_{f} P(f_{i,j}) \log \frac{P(f_{i,j})}{P(f_i)P(f_j)}$$ (Bu formül, doğrudan bir kod metodunda bulunmamaktadır, kavramsal bir metrik veya gelecekteki bir ekleme olabilir.)

### Statistical Validation

#### Effect Size (Cohen's d)
$$d = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2 + s_2^2}{2}}}$$ (Bu formül, genel istatistiksel doğrulama için kavramsal bir referanstır; doğrudan kodda bir metodla temsil edilmemektedir.)

#### Confidence Intervals (Bootstrap)
$$\text{CI}_{95\%} = [\text{percentile}_{2.5}, \text{percentile}_{97.5}]$$ (Bu formül, genel istatistiksel doğrulama için kavramsal bir referanstır; doğrudan kodda bir metodla temsil edilmemektedir.)

#### Significance Testing
$$t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}}$$ (Bu formül, genel istatistiksel doğrulama için kavramsal bir referanstır; doğrudan kodda bir metodla temsil edilmemektedir.)

### Performance Metrics

#### Computational Complexity
- NN-EEG: $O(L \cdot W \cdot \log W)$ per analysis
- NN-fMRI: $O(L \cdot G \cdot K)$ per ζ-score calculation
- Integration: $O(L \cdot (W + G))$ per validation

(Bu metrikler, kodun karmaşıklık analizini yansıtır ancak doğrudan bir metodla temsil edilmez.)

#### Memory Requirements
- NN-EEG: $O(L \cdot W)$ for temporal buffers  
- NN-fMRI: $O(L \cdot G)$ for spatial grids
- Total: $O(L \cdot (W + G))$

(Bu metrikler, kodun bellek kullanım analizini yansıtır ancak doğrudan bir metodla temsil edilmez.)

### Theoretical Guarantees

#### Convergence Properties
For ζ-score estimation:
$$\mathbb{E}[\hat{\zeta}(g)] = \zeta(g)$$
$$\text{Var}[\hat{\zeta}(g)] = O(1/K)$$

(Bu teoremler, `compute_zeta_scores` metodunun arkasındaki teorik garantileri ifade eder.)

#### Consistency Bounds
For cross-modal validation:
$$P(|\text{Consistency} - \text{True Consistency}| > \epsilon) \leq 2e^{-2n\epsilon^2}$$ 

(Bu teorem, `_cross_modal_validation_fixed` metodunun arkasındaki teorik garantiyi ifade eder.)

### Implementation Considerations

#### Numerical Stability
- Add $\epsilon = 10^{-8}$ to prevent $\log(0)$ (Bkz: `src/nn_neuroimaging/nn_fmri/implementation.py` - `__init__` ve `_calculate_activation_density`)
- Normalize PSD to prevent overflow (Bkz: `src/nn_neuroimaging/nn_eeg/implementation.py` - `analyze_frequency_domain` metodu)
- Use double precision for accumulations (Genel Python/PyTorch uygulaması, doğrudan belirli bir metoda bağlı değildir.)

#### Optimization Strategies
- Vectorized operations for efficiency (Genel PyTorch iyi uygulaması, doğrudan belirli bir metoda bağlı değildir.)
- Sparse grid representation for memory (Kavramsal, gelecekteki bir optimizasyon olabilir.)
- Incremental PSD updates for real-time (Kavramsal, gelecekteki bir optimizasyon olabilir.)

### Validation Metrics

#### Reproducibility
$$\text{CV} = \frac{\sigma}{\mu}$$ (Genel istatistiksel metrik, doğrudan bir metoda bağlı değildir.)
Target: CV < 0.05 for excellent reproducibility

#### Cross-Modal Consistency
$$\text{Cohen's } \kappa = \frac{p_o - p_e}{1 - p_e}$$ (Genel istatistiksel metrik, doğrudan bir metoda bağlı değildir.)
Target: κ > 0.8 for strong agreement

#### Statistical Power
$$\text{Power} = P(\text{reject } H_0 | H_1 \text{ true})$$ (Genel istatistiksel metrik, doğrudan bir metoda bağlı değildir.)
Target: Power > 0.8 for adequate detection

## Practical Applications (CIFAR-10 Case Study)

### NN-EEG Results
```python
# Sample output from CIFAR-10 analysis
{
  'layer1_conv': {
    'dominant_freq': 0.286, 
    'state': 'inference',
    'band_powers': {'gamma': 0.42, 'beta': 0.31}
  },
  'layer4_fc': {
    'dominant_freq': 0.143,
    'state': 'idle',
    'band_powers': {'alpha': 0.52}
  }
}
```

### NN-fMRI Results
```python
# Top 3 critical regions (ζ-scores)
[
  {'grid': (2,3,1), 'zeta': 8.7, 'p_value': 0.003},
  {'grid': (1,2,3), 'zeta': 7.2, 'p_value': 0.012},
  {'grid': (3,1,2), 'zeta': 6.8, 'p_value': 0.021}
]
```

## Cross-References
- See `framework-overview.md` for implementation status
- Refer to `proje_analiz_raporu.md` for validation metrics
- Check `nn-eeg-methodology.md` for temporal analysis details
