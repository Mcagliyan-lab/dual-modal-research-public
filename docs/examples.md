# Usage Examples

This section contains code examples demonstrating various usage scenarios of the Dual-Modal Neural Network Neuroimaging Framework.

## Quick Start Example

This example shows how to perform basic dual-modal analysis by integrating both NN-EEG and NN-fMRI modules.

Below is the code that runs the quick start example:

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

## Advanced Usage Examples

### 1. Custom Model Analysis

```python
from dual_modal.nn_eeg import NeuralEEG
from dual_modal.nn_fmri import NeuralFMRI
import torch
import torch.nn as nn

# Define your custom model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model and analyzers
model = CustomModel()
nn_eeg = NeuralEEG(model, sample_rate=10.0)
nn_fmri = NeuralFMRI(model, grid_size=(10, 10, 5))

# Perform dual-modal analysis
test_data = torch.randn(32, 3, 224, 224)
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data, torch.randint(0, 10, (32,))),
    batch_size=8
)

# NN-EEG Analysis
temporal_signals = nn_eeg.extract_temporal_signals(dataloader)
frequency_analysis = nn_eeg.analyze_frequency_domain(temporal_signals)
eeg_state = nn_eeg.classify_operational_states(frequency_analysis)

# NN-fMRI Analysis
fmri_results = nn_fmri.analyze_spatial_patterns(test_data)
zeta_scores = nn_fmri.compute_zeta_scores(test_data)

print(f"EEG State: {eeg_state}")
print(f"fMRI Spatial Patterns: {len(fmri_results)} regions analyzed")
```

### 2. Cross-Modal Validation

```python
from dual_modal.integration import DualModalIntegrator

# Initialize integrated analyzer
integrator = DualModalIntegrator(model)

# Perform comprehensive analysis
results = integrator.analyze(test_data, report_type='detailed')

# Cross-modal validation
validation_metrics = integrator.cross_modal_validation(
    results['nn_eeg'], 
    results['nn_fmri']
)

print(f"Cross-modal consistency: {validation_metrics['consistency_score']:.3f}")
print(f"Validation status: {validation_metrics['validation_status']}")
```

### 3. Batch Processing Pipeline

```python
import os
from pathlib import Path

def process_model_batch(model_paths, output_dir):
    """Process multiple models in batch"""
    
    for model_path in model_paths:
        print(f"Processing: {model_path}")
        
        # Load model
        model = torch.load(model_path)
        model.eval()
        
        # Create analyzers
        integrator = DualModalIntegrator(model)
        
        # Generate test data
        test_data = torch.randn(50, 3, 224, 224)
        
        # Perform analysis
        results = integrator.analyze(test_data)
        
        # Save results
        output_file = Path(output_dir) / f"{Path(model_path).stem}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")

# Example usage
model_paths = [
    'models/resnet18.pth',
    'models/vgg16.pth',
    'models/mobilenet.pth'
]

process_model_batch(model_paths, 'results/batch_analysis/')
```

### 4. Real-time Monitoring

```python
import time
import threading
from collections import deque

class RealtimeMonitor:
    def __init__(self, model, window_size=100):
        self.model = model
        self.nn_eeg = NeuralEEG(model)
        self.window_size = window_size
        self.signal_buffer = deque(maxlen=window_size)
        self.running = False
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            # Simulate incoming data
            batch_data = torch.randn(4, 3, 224, 224)
            
            # Extract signals
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(batch_data, torch.zeros(4)),
                batch_size=4
            )
            
            signals = self.nn_eeg.extract_temporal_signals(dataloader, max_batches=1)
            
            # Update buffer
            if signals:
                avg_signal = np.mean([np.mean(sig) for sig in signals.values()])
                self.signal_buffer.append(avg_signal)
            
            # Analyze current window
            if len(self.signal_buffer) >= 10:
                recent_signals = list(self.signal_buffer)[-10:]
                freq_analysis = self.nn_eeg.analyze_frequency_domain({'current': recent_signals})
                state = self.nn_eeg.classify_operational_states(freq_analysis)
                
                print(f"Current state: {state}, Signal strength: {avg_signal:.3f}")
            
            time.sleep(0.1)  # 10Hz monitoring
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

# Example usage
monitor = RealtimeMonitor(model)
monitor.start_monitoring()

# Let it run for 30 seconds
time.sleep(30)
monitor.stop_monitoring()
```

## Visualization Examples

### Creating Comprehensive Visualizations

```python
from dual_modal.visualization import create_comprehensive_visualizations

# After performing analysis
results = integrator.analyze(test_data)

# Create all visualizations
create_comprehensive_visualizations(
    results, 
    output_dir='visualizations/',
    include_3d=True,
    save_interactive=True
)

print("Visualizations saved to 'visualizations/' directory")
```

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Use GPU-accelerated data loading
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)
```

### Memory Optimization

```python
# For large models, use gradient checkpointing
from torch.utils.checkpoint import checkpoint

# Process in smaller batches to reduce memory usage
for batch_data, _ in dataloader:
    with torch.no_grad():
        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        outputs = model(batch_data)
```

These examples demonstrate the flexibility and power of the Dual-Modal Neural Network Neuroimaging Framework. Each example can be adapted to your specific research needs and extended with additional functionality.