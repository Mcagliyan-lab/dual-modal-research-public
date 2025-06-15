# Neural Network Architecture

## Overview

This document describes the detailed architecture of our dual-modal neural network system, including component specifications, data flow, and design decisions.

## System Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐
│   Image Input   │    │   Text Input    │
│   [B,3,224,224] │    │   [B,L]         │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Vision Encoder  │    │  Text Encoder   │
│   (ResNet-50)   │    │   (BERT-base)   │
│ [B,2048]        │    │ [B,L,768]       │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────┬─────────────┬─┘
                 ▼             ▼
         ┌─────────────────────────┐
         │  Cross-Modal Attention  │
         │      [B,512]            │
         └─────────┬───────────────┘
                   ▼
         ┌─────────────────────────┐
         │     Classifier          │
         │   [B,num_classes]       │
         └─────────────────────────┘
```

## Component Details

### 1. Vision Encoder

**Architecture**: ResNet-50 (pre-trained on ImageNet)
- **Input**: RGB images [batch_size, 3, 224, 224]
- **Output**: Feature vectors [batch_size, 2048]
- **Modifications**: Removed final classification layer

**Key Features**:
- Residual connections for gradient flow
- Batch normalization for training stability
- Pre-trained weights for transfer learning

### 2. Text Encoder

**Architecture**: BERT-base-uncased
- **Input**: Tokenized text sequences [batch_size, sequence_length]
- **Output**: Contextualized embeddings [batch_size, sequence_length, 768]
- **Max Length**: 512 tokens

**Key Features**:
- Bidirectional attention mechanism
- WordPiece tokenization
- Position embeddings
- Layer normalization

### 3. Cross-Modal Fusion Layer

**Architecture**: Multi-Head Cross-Attention
- **Heads**: 8 attention heads
- **Hidden Dimension**: 512
- **Dropout**: 0.1

**Attention Mechanism**:
```python
# Text-to-Visual Attention
Q_t = text_features @ W_q
K_v = visual_features @ W_k  
V_v = visual_features @ W_v
Attention_tv = softmax(Q_t @ K_v^T / √d_k) @ V_v

# Visual-to-Text Attention  
Q_v = visual_features @ W_q
K_t = text_features @ W_k
V_t = text_features @ W_v
Attention_vt = softmax(Q_v @ K_t^T / √d_k) @ V_t
```

**Fusion Process**:
1. Project features to common dimension
2. Compute bidirectional cross-attention
3. Pool attended features
4. Concatenate and project to output dimension

### 4. Classification Head

**Architecture**: Two-layer MLP
- **Layer 1**: Linear(512 → 256) + ReLU + Dropout(0.1)
- **Layer 2**: Linear(256 → num_classes)
- **Output**: Class logits [batch_size, num_classes]

## Data Flow

### Forward Pass Pipeline

1. **Image Processing**:
   ```
   Images → ResNet-50 → Global Average Pool → [B, 2048]
   ```

2. **Text Processing**:
   ```
   Text → Tokenization → BERT → [B, L, 768]
   ```

3. **Feature Projection**:
   ```
   Visual: [B, 2048] → Linear → [B, 1, 512]
   Text: [B, L, 768] → Linear → [B, L, 512]
   ```

4. **Cross-Modal Attention**:
   ```
   Visual ↔ Text → Multi-Head Attention → [B, 512]
   ```

5. **Classification**:
   ```
   Fused Features → MLP → [B, num_classes]
   ```

## Design Decisions

### Architecture Choices

**Why ResNet-50?**
- Proven performance on visual tasks
- Good balance between accuracy and efficiency
- Strong pre-trained representations

**Why BERT-base?**
- State-of-the-art text understanding
- Bidirectional context modeling
- Rich semantic representations

**Why Cross-Attention?**
- Learns meaningful cross-modal alignments
- Allows dynamic feature weighting
- Interpretable attention patterns

### Hyperparameter Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Fusion Dim | 512 | Balance between capacity and efficiency |
| Attention Heads | 8 | Standard transformer configuration |
| Dropout | 0.1 | Prevent overfitting without underfitting |
| Learning Rate | 1e-4 | Stable training for pre-trained models |

## Memory and Computational Requirements

### Training Requirements
- **GPU Memory**: ~12GB per GPU
- **Batch Size**: 32 (16 per GPU for multi-GPU)
- **Training Time**: ~2.3 hours on 8 V100 GPUs

### Inference Requirements
- **Latency**: 23ms per sample
- **Memory**: ~285MB model size
- **Throughput**: 43 samples/second

## Scalability Considerations

### Model Scaling
- **Larger Backbones**: Can use ResNet-101, EfficientNet, or ViT
- **Text Models**: Can upgrade to BERT-large or RoBERTa
- **Fusion Scaling**: Increase hidden dimensions and attention heads

### Efficiency Optimizations
- **Knowledge Distillation**: Train smaller student models
- **Quantization**: 8-bit or 16-bit precision
- **Pruning**: Remove less important connections

## Comparison with Alternatives

### Fusion Strategies

| Method | Pros | Cons | Performance |
|--------|------|------|-------------|
| Cross-Attention | Interpretable, flexible | Higher complexity | 94.2% |
| Concatenation | Simple, fast | Limited interaction | 91.5% |
| Element-wise | Efficient | Assumes alignment | 90.1% |

### Backbone Alternatives

| Vision Backbone | Accuracy | Parameters | Speed |
|----------------|----------|------------|-------|
| ResNet-50 | 94.2% | 25M | Fast |
| EfficientNet-B4 | 94.8% | 19M | Medium |
| ViT-Base | 95.1% | 86M | Slow |

## Future Architecture Improvements

### Short-term Enhancements
- **Adaptive Fusion**: Dynamic fusion weights based on input
- **Multi-Scale Features**: Use multiple ResNet layers
- **Attention Visualization**: Better interpretability tools

### Long-term Directions
- **Transformer-based Vision**: Replace CNN with Vision Transformer
- **Unified Architecture**: Single transformer for both modalities
- **Efficient Attention**: Linear attention mechanisms 