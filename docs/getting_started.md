# Getting Started

Welcome to the Dual-Modal Neural Network Neuroimaging Framework! This guide will help you get started with analyzing neural networks using our NN-EEG and NN-fMRI approaches.

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/research-lab/dual-modal-research-public.git
cd dual-modal-research-public

# Install dependencies
pip install -r requirements.txt

# For development dependencies (optional)
pip install -r requirements-dev.txt
```

## Quick Start

```python
# Run the basic example
python examples/quick_start.py

# For comprehensive analysis
python examples/full_analysis.py
```

## Basic Usage

### 1. Simple Analysis

```python
from dual_modal.integration import DualModalIntegrator
import torch

# Load your model
model = torch.load('path/to/your/model.pth')
model.eval()

# Create analyzer
analyzer = DualModalIntegrator(model)

# Prepare test data
test_data = torch.randn(32, 3, 224, 224)

# Perform dual-modal analysis
results = analyzer.analyze(test_data)

print(f"Analysis complete!")
print(f"Temporal patterns: {len(results['nn_eeg'])} layers analyzed")
print(f"Spatial patterns: {results['nn_fmri']['grid_size']} grid regions")
```

### 2. Individual Module Usage

```python
from dual_modal.nn_eeg import NeuralEEG
from dual_modal.nn_fmri import NeuralFMRI

# NN-EEG: Temporal analysis
nn_eeg = NeuralEEG(model, sample_rate=10.0)
temporal_results = nn_eeg.extract_temporal_signals(dataloader)
frequency_analysis = nn_eeg.analyze_frequency_domain(temporal_results)

# NN-fMRI: Spatial analysis  
nn_fmri = NeuralFMRI(model, grid_size=(8, 8, 4))
spatial_results = nn_fmri.analyze_spatial_patterns(test_data)
```

## Project Structure

```
dual-modal-research-public/
├── src/dual_modal/           # Main source code
│   ├── nn_eeg.py            # NN-EEG temporal analysis
│   ├── nn_fmri.py           # NN-fMRI spatial analysis
│   └── integration.py       # Dual-modal integration
├── examples/                # Usage examples
├── docs/                    # Documentation
├── tests/                   # Test suites
├── results/                 # Analysis outputs
└── requirements.txt         # Dependencies
```

## Next Steps

- **Understand the Theory**: Read our [research methodology](methodology.md) for theoretical background
- **Explore Examples**: Check out [usage examples](examples.md) for detailed code samples
- **API Reference**: Browse the [API documentation](api.md) for complete function details
- **Troubleshooting**: See [troubleshooting guide](troubleshooting.md) if you encounter issues

## Supported Models

The framework works with any PyTorch neural network, including:

- **Computer Vision**: ResNet, VGG, DenseNet, EfficientNet
- **Custom Architectures**: Any nn.Module-based model
- **Pre-trained Models**: Models from torchvision, timm, etc.

## System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.8.0 or higher
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional but recommended for large models

## Getting Help

- **Documentation**: Comprehensive guides in the `docs/` folder
- **Examples**: Working code samples in `examples/` 
- **Issues**: Report bugs or request features on GitHub
- **Community**: Join discussions in GitHub Discussions

---

Ready to start analyzing your neural networks? Jump into the [examples](examples.md) or explore our [methodology](methodology.md) to learn more about the science behind the framework! 