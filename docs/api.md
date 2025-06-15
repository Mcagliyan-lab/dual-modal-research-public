# API Reference

## DualModalNetwork

Main neural network class for dual-modal processing.

### Constructor

```python
DualModalNetwork(
    vision_backbone='resnet50',
    text_backbone='bert-base-uncased', 
    num_classes=1000,
    fusion_dim=512,
    num_heads=8
)
```

**Parameters:**
- `vision_backbone` (str): Vision encoder backbone ('resnet50')
- `text_backbone` (str): Text encoder backbone ('bert-base-uncased')
- `num_classes` (int): Number of output classes
- `fusion_dim` (int): Fusion layer hidden dimension
- `num_heads` (int): Number of attention heads

### Methods

#### forward(images, texts)
Forward pass through the network.

**Parameters:**
- `images` (torch.Tensor): Input images [B, 3, 224, 224]
- `texts` (list): List of text strings

**Returns:**
- `torch.Tensor`: Class logits [B, num_classes]

#### predict(images, texts)
Make predictions with softmax probabilities.

**Parameters:**
- `images` (torch.Tensor): Input images [B, 3, 224, 224]
- `texts` (list): List of text strings

**Returns:**
- `torch.Tensor`: Class probabilities [B, num_classes]

## Helper Functions

### create_model(config=None)
Factory function to create a dual-modal network.

**Parameters:**
- `config` (dict, optional): Model configuration

**Returns:**
- `DualModalNetwork`: Configured model instance

## Example Usage

```python
from models.dual_modal_net import create_model
import torch

# Create model with custom config
config = {
    'num_classes': 10,
    'fusion_dim': 256
}
model = create_model(config)

# Prepare inputs
images = torch.randn(2, 3, 224, 224)
texts = ["A cat", "A dog"]

# Forward pass
logits = model(images, texts)
predictions = model.predict(images, texts)
``` 