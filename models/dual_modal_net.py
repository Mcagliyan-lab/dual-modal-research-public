"""
Dual-Modal Neural Network Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing visual and text features."""
    
    def __init__(self, visual_dim, text_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, visual_features, text_features):
        # Project features to common dimension
        v_proj = self.visual_proj(visual_features)  # [B, H*W, hidden_dim]
        t_proj = self.text_proj(text_features)      # [B, L, hidden_dim]
        
        # Cross-modal attention: text attends to visual
        t_attended, _ = self.multihead_attn(
            query=t_proj,
            key=v_proj,
            value=v_proj
        )
        
        # Cross-modal attention: visual attends to text
        v_attended, _ = self.multihead_attn(
            query=v_proj,
            key=t_proj,
            value=t_proj
        )
        
        # Pool attended features
        t_pooled = t_attended.mean(dim=1)  # [B, hidden_dim]
        v_pooled = v_attended.mean(dim=1)  # [B, hidden_dim]
        
        # Concatenate and project
        fused = torch.cat([v_pooled, t_pooled], dim=-1)
        output = self.output_proj(fused)
        output = self.dropout(output)
        
        return output


class DualModalNetwork(nn.Module):
    """Dual-modal neural network for processing visual and textual data."""
    
    def __init__(self, 
                 vision_backbone='resnet50',
                 text_backbone='bert-base-uncased',
                 num_classes=1000,
                 fusion_dim=512,
                 num_heads=8):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Vision encoder
        if vision_backbone == 'resnet50':
            self.vision_encoder = models.resnet50(pretrained=True)
            vision_dim = self.vision_encoder.fc.in_features
            self.vision_encoder.fc = nn.Identity()  # Remove final FC layer
        else:
            raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(text_backbone)
        text_dim = self.text_encoder.config.hidden_size
        
        # Cross-modal fusion
        self.fusion_layer = CrossModalAttention(
            visual_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=fusion_dim,
            num_heads=num_heads
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
    def encode_images(self, images):
        """Encode images using vision backbone."""
        # Extract features from vision encoder
        features = self.vision_encoder(images)  # [B, vision_dim]
        
        # Reshape for attention (simulate spatial features)
        batch_size = features.size(0)
        features = features.unsqueeze(1)  # [B, 1, vision_dim]
        
        return features
    
    def encode_texts(self, texts):
        """Encode texts using text backbone."""
        # Tokenize texts
        if isinstance(texts[0], str):
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to same device as model
            device = next(self.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
        else:
            encoded = texts
        
        # Get text embeddings
        outputs = self.text_encoder(**encoded)
        text_features = outputs.last_hidden_state  # [B, L, text_dim]
        
        return text_features
    
    def forward(self, images, texts):
        """Forward pass through dual-modal network."""
        # Encode modalities
        visual_features = self.encode_images(images)
        text_features = self.encode_texts(texts)
        
        # Cross-modal fusion
        fused_features = self.fusion_layer(visual_features, text_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict(self, images, texts):
        """Make predictions on input data."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, texts)
            predictions = F.softmax(logits, dim=-1)
        return predictions


def create_model(config=None):
    """Factory function to create dual-modal network."""
    if config is None:
        config = {
            'vision_backbone': 'resnet50',
            'text_backbone': 'bert-base-uncased',
            'num_classes': 1000,
            'fusion_dim': 512,
            'num_heads': 8
        }
    
    return DualModalNetwork(**config)


if __name__ == "__main__":
    # Example usage
    model = create_model()
    
    # Dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["A cat sitting on a table"] * batch_size
    
    # Forward pass
    logits = model(images, texts)
    print(f"Output shape: {logits.shape}")
    
    # Predictions
    predictions = model.predict(images, texts)
    print(f"Predictions shape: {predictions.shape}") 