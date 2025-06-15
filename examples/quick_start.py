"""
Quick Start Example for Dual-Modal Neural Network
"""

import torch
from models.dual_modal_net import create_model


def main():
    """Quick start example demonstrating basic usage."""
    
    print("🚀 Dual-Modal Neural Network - Quick Start")
    print("=" * 50)
    
    # Create model
    print("📦 Creating dual-modal model...")
    model = create_model({
        'vision_backbone': 'resnet50',
        'text_backbone': 'bert-base-uncased',
        'num_classes': 10,  # Example: 10 classes
        'fusion_dim': 512,
        'num_heads': 8
    })
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Prepare sample data
    print("\n📊 Preparing sample data...")
    batch_size = 4
    
    # Sample images (random tensors for demo)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Sample texts
    texts = [
        "A cat sitting on a wooden table",
        "A dog running in the park",
        "A bird flying in the sky",
        "A car driving on the road"
    ]
    
    print(f"✅ Prepared {batch_size} image-text pairs")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        predictions = model.predict(images, texts)
        
        # Get raw logits
        logits = model(images, texts)
    
    print(f"✅ Forward pass completed")
    print(f"📈 Output shape: {logits.shape}")
    print(f"🎯 Predictions shape: {predictions.shape}")
    
    # Display results
    print("\n📋 Sample Results:")
    print("-" * 30)
    
    for i, text in enumerate(texts):
        top_prob = predictions[i].max().item()
        top_class = predictions[i].argmax().item()
        
        print(f"Text: '{text[:30]}...'")
        print(f"  → Top class: {top_class} (confidence: {top_prob:.3f})")
        print()
    
    print("🎉 Quick start completed successfully!")
    
    # Model info
    print("\n📊 Model Information:")
    print(f"  • Vision Encoder: ResNet-50")
    print(f"  • Text Encoder: BERT-base")
    print(f"  • Fusion Method: Cross-Modal Attention")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Model Size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main() 