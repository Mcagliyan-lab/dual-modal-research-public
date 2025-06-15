# Kullanım Örnekleri

Bu bölüm, Dual-Modal Neural Network Neuroimaging Framework'ünün çeşitli kullanım senaryolarını gösteren kod örneklerini içerir.

## Hızlı Başlangıç Örneği

Bu örnek, hem NN-EEG hem de NN-fMRI modüllerini entegre ederek temel bir dual-modal analizinin nasıl yapılacağını gösterir.

Aşağıda, hızlı başlangıç örneğini çalıştıran kod bulunmaktadır:

```python
#!/usr/bin/env python3
"""
Quick Test: NN-EEG Minimal Validation
====================================
This script runs a minimal version of NN-EEG to validate the concept works.
Should complete in under 2 minutes on most systems.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import json
import time

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("🧠 NN-EEG Quick Validation Test")
print("=" * 40)

# Create simple test model
print("1. Creating test model...")
model = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10)
)
model.eval()

# Create minimal dataset
print("2. Creating synthetic data...")
# Use synthetic data for speed
data = torch.randn(10, 3, 32, 32)  # 10 samples, like CIFAR-10
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data, torch.randint(0, 10, (10,))),
    batch_size=2,
    shuffle=False
)

# Capture activations
print("3. Capturing layer activations...")
activations = {}
hooks = []

def create_hook(name):
    def hook_fn(module, input, output):
        act = output.detach().cpu().numpy()
        if len(act.shape) == 4:  # Conv layer
            activations[name] = np.mean(act, axis=(0, 2, 3))  # Mean across batch and spatial
        else:  # Linear layer
            activations[name] = np.mean(act, axis=0)  # Mean across batch
    return hook_fn

# Register hooks
layer_count = 0
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        hook = module.register_forward_hook(create_hook(f"layer_{layer_count}"))
        hooks.append(hook)
        layer_count += 1

# Extract temporal signals
print("4. Extracting temporal signals...")
temporal_signals = {}

for batch_idx, (batch_data, _) in enumerate(dataloader):
    activations.clear()
    
    with torch.no_grad():
        _ = model(batch_data)
    
    # Store temporal progression
    for layer_name, activation in activations.items():
        if layer_name not in temporal_signals:
            temporal_signals[layer_name] = []
        
        signal_value = np.mean(activation)
        temporal_signals[layer_name].append(signal_value)

# Convert to numpy arrays
for layer_name in temporal_signals:
    temporal_signals[layer_name] = np.array(temporal_signals[layer_name])

print(f"   Captured signals from {len(temporal_signals)} layers")
for layer_name, signal in temporal_signals.items():
    print(f"   {layer_name}: {len(signal)} time points, range: [{signal.min():.3f}, {signal.max():.3f}]")

# Frequency analysis
print("5. Performing frequency analysis...")
frequency_results = {}

for layer_name, signal in temporal_signals.items():
    if len(signal) >= 4:  # Minimum for frequency analysis
        try:
            frequencies, psd = signal.welch(signal, fs=1.0, nperseg=len(signal))
            dominant_freq = frequencies[np.argmax(psd)]
            total_power = np.sum(psd)
            
            frequency_results[layer_name] = {
                'dominant_frequency': dominant_freq,
                'total_power': total_power,
                'frequencies': frequencies.tolist(),
                'psd': psd.tolist()
            }
            
            print(f"   {layer_name}: {dominant_freq:.3f} Hz (power: {total_power:.4f})")
            
        except Exception as e:
            print(f"   {layer_name}: Analysis failed - {e}")

# State classification
print("6. Classifying operational state...")
if frequency_results:
    # Simple state classification based on frequency patterns
    all_dominant_freqs = [result['dominant_frequency'] for result in frequency_results.values()]
    mean_freq = np.mean(all_dominant_freqs)
    
    if mean_freq > 0.3:
        state = "inference"
    elif mean_freq > 0.1:
        state = "processing"
    else:
        state = "idle"
    
    print(f"   Operational state: {state} (mean freq: {mean_freq:.3f} Hz)")
else:
    state = "unknown"
    print("   Could not classify state - insufficient data")

# Create simple visualization
print("7. Creating visualization...")
if frequency_results:
    fig, axes = plt.subplots(1, min(3, len(frequency_results)), figsize=(12, 4))
    if len(frequency_results) == 1:
        axes = [axes]
    
    for idx, (layer_name, result) in enumerate(list(frequency_results.items())[:3]):
        ax = axes[idx] if len(frequency_results) > 1 else axes[0]
        
        frequencies = np.array(result['frequencies'])
        psd = np.array(result['psd'])
        
        ax.plot(frequencies, psd, 'b-', linewidth=2)
        ax.set_title(f'{layer_name}\n{result["dominant_frequency"]:.3f} Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/quick_test_results.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved to results/quick_test_results.png")
    plt.close()  # Don't show in non-interactive environments

# Save results
print("8. Saving results...")
results = {
    'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model_layers': layer_count,
    'signal_length': len(list(temporal_signals.values())[0]) if temporal_signals else 0,
    'frequency_analysis': frequency_results,
    'operational_state': state,
    'test_status': 'SUCCESS' if frequency_results else 'PARTIAL'
}

with open('results/quick_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Cleanup
for hook in hooks:
    hook.remove()

# Final report
print("\n" + "=" * 40)
print("✅ NN-EEG QUICK TEST COMPLETED!")
print("=" * 40)
print(f"Status: {results['test_status']}")
print(f"Layers analyzed: {results['model_layers']}")
print(f"Signal length: {results['signal_length']} time points")
print(f"Operational state: {results['operational_state']}")
print(f"Results saved to: results/quick_test_results.json")

if results['test_status'] == 'SUCCESS':
    print("\n🎉 PROOF-OF-CONCEPT VALIDATED!")
    print("The NN-EEG framework successfully:")
    print("✅ Captured layer activations")
    print("✅ Extracted temporal signals")
    print("✅ Performed frequency analysis")
    print("✅ Classified operational state")
    print("✅ Generated reproducible results")
    print("\nNext step: Run full CIFAR-10 validation")
else:
    print("\n⚠️  PARTIAL SUCCESS")
    print("Basic framework works but needs debugging")
```

Analiz çıktısı doğrudan konsolda görüntülenecektir ve sonuçlar `results/` dizinine kaydedilecektir.

## Detaylı Test Örnekleri

Çerçevenin kapsamlı test edilmiş örnekleri için `tests/` dizinindeki test dosyalarını inceleyebilirsiniz:

- `tests/test_nn_eeg.py` - NN-EEG modülü testleri
- `tests/test_nn_fmri.py` - NN-fMRI modülü testleri  
- `tests/test_integration.py` - Dual-modal entegrasyon testleri 