# Implementation Guide

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from models.dual_modal_net import create_model

# Initialize model
model = create_model()

# Load data
import torch
images = torch.randn(4, 3, 224, 224)
texts = ["Sample text description"] * 4

# Predict
predictions = model.predict(images, texts)
```

## Model Architecture

### Core Components
- **Vision Encoder**: CNN-based feature extraction
- **Text Encoder**: Transformer-based text processing  
- **Fusion Layer**: Cross-modal attention mechanism
- **Classifier**: Final prediction layer

### Configuration
```python
config = {
    'vision_backbone': 'resnet50',
    'text_backbone': 'bert-base-uncased',
    'num_classes': 1000,
    'fusion_dim': 512,
    'num_heads': 8,
    'dropout': 0.1
}

model = create_model(config)
```

## Training

### Data Format
```python
# Expected input format
images = torch.tensor([batch_size, 3, 224, 224])  # RGB images
texts = ['text sample 1', 'text sample 2', ...]   # List of strings
labels = torch.tensor([batch_size])               # Class labels
```

### Training Loop
```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = create_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training
model.train()
for epoch in range(num_epochs):
    for images, texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Evaluation

### Metrics
- Accuracy
- F1-Score  
- Precision/Recall
- Cross-modal attention analysis

### Example
```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, texts, labels in test_loader:
        outputs = model(images, texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
```

## Advanced Usage

### Custom Model Configuration
```python
# Custom configuration
config = {
    'vision_backbone': 'resnet50',
    'text_backbone': 'bert-base-uncased',
    'num_classes': 10,
    'fusion_dim': 256,
    'num_heads': 4,
    'dropout': 0.2
}

model = create_model(config)
```

### Attention Visualization
```python
# Get attention weights
model.eval()
with torch.no_grad():
    outputs, attention_weights = model.forward_with_attention(images, texts)
    
# Visualize attention patterns
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0].cpu().numpy())
plt.title('Cross-Modal Attention')
plt.show()
``` 