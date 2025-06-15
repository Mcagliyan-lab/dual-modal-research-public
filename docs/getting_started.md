# Getting Started

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/Mcagliyan-lab/dual-modal-research-public.git
cd dual-modal-research-public

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Run the example
python examples/quick_start.py
```

## Basic Usage

```python
from models.dual_modal_net import create_model
import torch

# Create model
model = create_model()

# Prepare data
images = torch.randn(4, 3, 224, 224)
texts = ["Sample text description"] * 4

# Get predictions
predictions = model.predict(images, texts)
```

## Next Steps

- Read the [methodology](methodology.md) for theoretical background
- Check out more examples in the `examples/` directory
- Explore the model implementation in `models/dual_modal_net.py` 