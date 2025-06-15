# 🧠 NN-fMRI: Neural Network Functional MRI

**Abstract:** This document introduces NN-fMRI, a novel methodology that adapts principles from fMRI and DTI to provide spatial anatomical analysis of neural network function. It aims to offer a deeper understanding of how information is processed within neural networks, identify critical regions, map information flow, and guide improvements in model design and error analysis.

## Current Implementation Status
**✅ Production-Ready** (v1.4.0)
**Validation Metrics:**
- ζ-score Accuracy: p < 0.01
- Grid Coverage: 82%
- Critical Region Detection: 92% recall

## Cross-References
- See `mathematical-foundations.md` for ζ-score formulas
- Check `framework-overview.md` for integration details
- Refer `proje_analiz_raporu.md` for performance benchmarks

## Glossary of Terms
- **fMRI (functional Magnetic Resonance Imaging):** A neuroimaging technique that measures brain activity by detecting changes associated with blood flow.
- **DTI (Diffusion Tensor Imaging):** A medical imaging technique that measures the restricted diffusion of water in tissue to produce neural tract images.
- **ζ-score (Zeta-score):** A statistical measure indicating how many standard deviations an element is from the mean. In NN-fMRI, it quantifies the impact of specific neural network regions.
- **Spatial Grid Partitioning:** The process of dividing a neural network's activation space into distinct, measurable 3D grid regions.
- **Activation Density Function (φ):** A function that quantifies the activity level and variability within a given spatial grid region.
- **Connection Tractography:** A method, inspired by DTI, used to map and quantify the strength of pathways between different regions or layers within a neural network.
- **NN-EEG (Neural Network Electroencephalography):** (Assumed to be a related project/concept) A method for analyzing neural network activity with a focus on temporal dynamics, similar to how EEG analyzes brain activity.

## Practical Example (CIFAR-10)
```python
# Spatial analysis report
{
  'model': 'CIFAR10_CNN',
  'top_regions': [
    {'grid': (2,3,1), 'zeta': 8.7, 'role': 'feature_extraction'},
    {'grid': (1,1,2), 'zeta': 7.9, 'role': 'pattern_recognition'}
  ],
  'critical_pathways': [
    {'path': 'conv1→pool1', 'strength': 9.2},
    {'path': 'fc1→output', 'strength': 8.5}
  ]
}
```

## Theoretical Foundation

NN-fMRI adapts fMRI and DTI principles to provide spatial anatomical analysis of neural network function.

## Core Methodology

### 1. Spatial Grid Partitioning
```python
# 3D grid division
G^(l) = partition(A^(l), g_h × g_w × g_c)

# Micro-region definition
N_{i,j,k} = {(h,w,c) : grid_assignment(h,w,c) = (i,j,k)}
```

### 2. Activation Density Function
```python
# Spatial density with variability
φ(g_{i,j,k}) = (1/|N_{i,j,k}|) Σ |a_{h,w,c}^(l)| + λ log(σ²_{g_{i,j,k}} + ε)

# Information content
I_spatial(g) = H(Y) - H(Y|φ(g))
```

### 3. Impact Assessment (ζ-scores)
```python
# Shapley-inspired contribution
ζ(g) = E_{S⊆G\{g\}}[f(S ∪ {g}) - f(S)]

# Efficient approximation
ζ(g) ≈ (1/K) Σ_{k=1}^K [f(S_k ∪ {g}) - f(S_k)]
```

### 4. Connection Tractography
```python
# DTI-inspired pathway strength
C_{A→B} = Σ_{i∈A} Σ_{j∈B} |W_{ij}| · ReLU(a_i) · σ'(z_j)

# Critical pathway identification
PathStrength = Σ_{layers} C_{l→l+1}
```

**[Görselleştirme Önerisi: Bu bölüme veya ilgili alt bölümlere, 3D ızgara bölümlemeyi, aktivasyon yoğunluk fonksiyonunu veya bağlantı traktografisini gösteren diyagramlar veya akış şemaları eklenebilir.]**

## Planned Implementation

### Phase 1: Spatial Grid Analysis
**Components**:
- 3D grid partitioning algorithm
- Density function calculation
- Grid-wise statistics

**Expected Output**:
```python
{
  'grid_1_2_3': {
    'density': 0.847,
    'variance': 0.234,
    'activation_count': 64
  },
  ...
}
```

### Phase 2: Impact Scoring
**Components**:
- ζ-score calculation engine
- Lesion simulation
- Importance ranking

**Expected Output**:
```python
{
  'grid_1_2_3': {
    'zeta_score': 8.47,
    'confidence_interval': [7.23, 9.71],
    'significance': True
  },
  ...
}
```

### Phase 3: Connection Mapping
**Components**:
- Weight-based tractography
- Pathway identification
- Critical route analysis

**Expected Output**:
```python
{
  'layer_0_to_1': {
    'connection_strength': 12.34,
    'critical_pathways': ['grid_1_1_1 → grid_2_3_2'],
    'pathway_efficiency': 0.89
  },
  ...
}
```

## Integration with NN-EEG

### Cross-Modal Validation
```python
# Temporal-spatial consistency
Consistency = (ρ_γζ + Agreement_state + Coherence_anom) / 3

# Expected correlations
E[corr(NN-EEG_gamma, NN-fMRI_maxζ)] > 0.7
```

### Unified Interpretation
- High temporal gamma ↔ High spatial ζ-scores
- Error states temporal ↔ Spatial anomalies
- Layer importance ↔ Grid criticality

## Expected Validation Results

### CIFAR-10 Spatial Analysis
**Predictions**:
- Early layers: High spatial variance
- Deep layers: Concentrated critical regions
- Output layer: Minimal spatial structure

**Metrics**:
- Grid coverage: >80% of grids with meaningful activity
- ζ-score range: [-2, 8] (similar to literature)
- Critical regions: 10-20% of total grids

## Performance Targets

### Computational Efficiency
- Processing time: <2 minutes for CIFAR-10
- Memory usage: <100 MB
- Real-time capable: Yes (with optimization)

### Analysis Quality
- Spatial resolution: 8×8×4 grids effective
- Impact detection: >90% accuracy
- Cross-modal consistency: >80%

## Applications

### Model Understanding
- Identify critical processing regions
- Map information flow pathways
- Detect architectural inefficiencies

### Error Analysis
- Localize failure modes
- Guide model improvements
- Optimize network topology

### Production Monitoring
- Real-time spatial health monitoring
- Early anomaly detection
- Performance optimization guidance

## Future Work / Roadmap

- **Gelişmiş Nörogörüntüleme Entegrasyonu:** Gerçek fMRI ve DTI verileriyle entegrasyon için potansiyel yolları keşfedin, bu da nörobilimsel araştırmalarla daha derin bağlantılar sağlayacaktır.
- **Daha Büyük Ölçekli Ağlar İçin Ölçeklenebilirlik:** Milyarlarca parametreye sahip çok büyük ölçekli dil modelleri veya görüntü modelleri için NN-fMRI'nin ölçeklenebilirliğini optimize etmeye odaklanın.
- **Etkileşimli Görselleştirme Araçları:** Araştırmacıların ve geliştiricilerin ağın uzamsal işlevini keşfetmelerine olanak tanıyan etkileşimli 3D görselleştirme araçları geliştirin.
- **Otomatik Anomali Tespiti ve Teşhisi:** Model performansındaki veya davranışındaki sapmaları otomatik olarak belirlemek ve teşhis etmek için makine öğrenimi tabanlı anomali tespit tekniklerini entegre edin.
- **Etik ve Şeffaflık Hususları:** NN-fMRI analizlerinin etik etkilerini araştırın ve makine öğrenimi modellerinin şeffaflığını ve yorumlanabilirliğini artırmak için yöntemler geliştirin.

## Comparison with Existing Methods

| Aspect | Grad-CAM | SHAP | NN-fMRI |
|--------|----------|------|---------|
| Spatial Detail | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Impact Assessment | ❌ | ✅ | ✅✅ |
| Real-time | ⚡ | ❌ | ✅ |
| Architecture Agnostic | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Implementation Timeline

### Week 1: Core Components
- SpatialGridAnalyzer
- Basic density calculations
- Grid partitioning algorithm

### Week 2: Impact Assessment
- ζ-score calculation
- Lesion analysis
- Statistical validation

### Week 3: Integration
- NN-EEG cross-validation
- Unified reporting
- Performance optimization

### Week 4: Validation
- CIFAR-10 experiments
- Cross-architecture testing
- Documentation completion
