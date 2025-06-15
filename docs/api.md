# 📖 API Documentation

## NN-EEG Module

### NeuralEEG Class

**Main temporal analysis class**

#### `__init__(model, sample_rate=1.0)`
Initialize NN-EEG analyzer
- `model`: PyTorch neural network
- `sample_rate`: Sampling rate for frequency analysis

#### `extract_temporal_signals(dataloader, max_batches=50)`
Extract temporal signals from layer activations
- Returns: Dictionary of layer-wise temporal signals

#### `analyze_frequency_domain(temporal_signals=None)`
Perform frequency domain analysis
- Returns: Frequency analysis results with PSD and band powers

#### `classify_operational_states(frequency_analysis=None)`
Classify network operational state
- Returns: State classification ('training', 'inference', 'idle', 'error')

## NN-fMRI Module

### NeuralFMRI Class

**Main spatial analysis class**

#### `__init__(model, grid_size=(8,8,4))`
Initialize NN-fMRI analyzer
- `model`: PyTorch neural network  
- `grid_size`: 3D grid dimensions for spatial partitioning

#### `analyze_spatial_patterns(data)`
Analyze spatial activation patterns
- Returns: Spatial analysis results

#### `compute_zeta_scores(validation_data)`
Compute impact scores for spatial regions
- Returns: ζ-scores for each grid region

## Integration Module

### DualModalIntegrator Class

**Combined temporal + spatial analysis**

#### `__init__(model)`
Initialize dual-modal analyzer

#### `analyze(data, report_type='technical')`
Complete dual-modal analysis
- Returns: Comprehensive analysis results

#### `cross_modal_validation(nn_eeg_results, nn_fmri_results)`
Perform cross-modal validation between NN-EEG and NN-fMRI results
- Returns: Validation metrics and consistency scores

## Utility Modules

### Data Loaders
- `create_cifar10_dataloader()`: CIFAR-10 dataset loader
- `create_mnist_dataloader()`: MNIST dataset loader  
- `create_synthetic_dataloader()`: Synthetic test data loader

### Visualization
- `plot_frequency_analysis()`: NN-EEG results visualization
- `plot_spatial_grids()`: NN-fMRI results visualization
- `create_comprehensive_visualizations()`: Combined visualization dashboard

### Metrics
- `cross_modal_validation()`: Cross-modal consistency evaluation
- `compute_consistency_score()`: Dual-modal consistency metrics
