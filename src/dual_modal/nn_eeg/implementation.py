"""
Neural Network Electroencephalography (NN-EEG) Implementation
=============================================================

Core implementation of temporal dynamics analysis for neural networks.
This is a working proof-of-concept that demonstrates the theoretical framework.

Author: Independent Research
Date: December 22, 2024
Version: 0.1.0 (Minimal Viable Implementation)

IMPORTANT: This implements the theoretical framework described in our paper.
           Results are preliminary but reproducible.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
from nn_neuroimaging.utils.config_utils import load_config
warnings.filterwarnings('ignore')

class ActivationCapture:
    """
    Captures layer activations during forward pass for temporal analysis.
    This is the foundation of our NN-EEG approach.
    """
    
    def __init__(self, model: nn.Module, layer_types: tuple = (nn.Conv2d, nn.Linear)):
        self.model = model
        self.layer_types = layer_types
        self.activations = {}
        self.hooks = []
        self.temporal_buffer = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layer types"""
        def create_hook(name):
            def hook_fn(module, input, output):
                # Convert to numpy and aggregate spatial dimensions
                activation = output.detach().cpu().numpy()
                
                # Aggregate based on tensor dimensions
                if len(activation.shape) == 4:  # Conv layers (B, C, H, W)
                    aggregated = np.mean(activation, axis=(0, 2, 3))  # Keep channels
                elif len(activation.shape) == 2:  # Linear layers (B, F)
                    aggregated = np.mean(activation, axis=0)  # Keep features
                else:
                    aggregated = np.mean(activation)  # Fallback to scalar
                
                self.activations[name] = aggregated
            return hook_fn
        
        # Register hooks on target layers
        layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types):
                hook = module.register_forward_hook(create_hook(f"layer_{layer_count}_{name}"))
                self.hooks.append(hook)
                layer_count += 1
                
        print(f"Registered hooks on {layer_count} layers")
    
    def capture_batch_activations(self, data_batch: torch.Tensor) -> Dict[str, np.ndarray]:
        """Capture activations for a single batch"""
        self.activations.clear()
        
        with torch.no_grad():
            _ = self.model(data_batch)
        
        return dict(self.activations)
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class NeuralEEG:
    """
    Main NN-EEG analyzer implementing temporal dynamics analysis.
    
    This class implements the core concepts from our theoretical framework:
    - Temporal signal extraction from layer activations
    - Frequency domain analysis using Welch's method
    - Operational state classification based on spectral patterns
    """
    
    def __init__(self, model: nn.Module, config_path: str = "config.yaml"):
        self.model = model
        self.device = next(model.parameters()).device
        
        # Load configuration
        self.config = load_config(Path(config_path))
        eeg_config = self.config['model_parameters']['nn_eeg']
        analysis_settings = self.config['analysis_settings']['general']
        
        self.sample_rate = eeg_config.get('sample_rate', 1.0)
        self.activation_capture = ActivationCapture(model)
        
        # Frequency bands (adapted from EEG neuroscience)
        self.frequency_bands = {
            'delta': (0.5, 4),     # Deep processing
            'theta': (4, 8),       # Memory/learning
            'alpha': (8, 13),      # Idle states  
            'beta': (13, 30),      # Active processing
            'gamma': (30, 100)     # High-level cognition
        }
        
        # Analysis parameters from config
        self.window_size = eeg_config.get('window_size', 50)
        self.overlap_ratio = eeg_config.get('overlap_ratio', 0.5)
        self.max_batches_eeg = analysis_settings.get('max_batches_eeg', 50)
        
        # Results storage
        self.temporal_signals = {}
        self.frequency_analysis = {}
        self.state_classifications = []
        
    def extract_temporal_signals(self, dataloader) -> Dict[str, np.ndarray]:
        """
        Extract temporal signals from neural network activations.
        
        This is the core NN-EEG process: treating layer activations as 
        time series data for frequency analysis.
        """
        print(f"Extracting temporal signals from {self.max_batches_eeg} batches...")
        
        temporal_data = {}
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if batch_count >= self.max_batches_eeg:
                break
                
            # Capture activations for this batch
            activations = self.activation_capture.capture_batch_activations(data.to(self.device))
            
            # Store temporal progression
            for layer_name, activation in activations.items():
                if layer_name not in temporal_data:
                    temporal_data[layer_name] = []
                
                # Convert to scalar signal (mean of all neurons/channels)
                if isinstance(activation, np.ndarray):
                    signal_value = np.mean(activation)
                else:
                    signal_value = float(activation)
                    
                temporal_data[layer_name].append(signal_value)
            
            batch_count += 1
            
            # Progress indicator
            if batch_count % 10 == 0:
                print(f"  Processed {batch_count} batches...")
        
        # Convert lists to numpy arrays
        for layer_name in temporal_data:
            temporal_data[layer_name] = np.array(temporal_data[layer_name])
        
        processing_time = time.time() - start_time
        print(f"Temporal signal extraction completed in {processing_time:.2f}s")
        
        self.temporal_signals = temporal_data
        return temporal_data
    
    def analyze_frequency_domain(self, temporal_signals: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
        """
        Perform frequency domain analysis using Welch's method.
        
        This implements the spectral analysis core of NN-EEG methodology.
        """
        if temporal_signals is None:
            temporal_signals = self.temporal_signals
            
        if not temporal_signals:
            raise ValueError("No temporal signals available. Run extract_temporal_signals first.")
        
        print("Performing frequency domain analysis...")
        
        frequency_results = {}
        
        for layer_name, layer_signal in temporal_signals.items():
            if len(layer_signal) < 10:  # Skip layers with insufficient data
                continue
                
            try:
                # Apply Welch's method for power spectral density
                nperseg = int(self.window_size)  # Use config for nperseg
                noverlap = int(self.window_size * self.overlap_ratio) # Use config for noverlap

                if nperseg > len(layer_signal) or noverlap > len(layer_signal):
                    warnings.warn(f"Not enough data for Welch's method for layer {layer_name}. Adjusting nperseg/noverlap.")
                    nperseg = len(layer_signal) // 2
                    noverlap = nperseg // 2
                
                if nperseg < 4:
                    nperseg = min(len(layer_signal), 4) # Ensure a minimum of 4 for nperseg
                    noverlap = nperseg // 2

                frequencies, psd = signal.welch(
                    layer_signal, 
                    fs=self.sample_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling='density'
                )
                
                # Ensure psd is not empty before processing
                if len(psd) == 0:
                    warnings.warn(f"Empty PSD for layer {layer_name}. Skipping frequency analysis.")
                    continue

                # Compute band powers
                band_powers = self._extract_band_powers(frequencies, psd)
                
                # Compute spectral features
                spectral_features = self._compute_spectral_features(frequencies, psd)
                
                # Dominant frequency (peak in PSD)
                dominant_frequency = frequencies[np.argmax(psd)]
                
                frequency_results[layer_name] = {
                    'frequencies': frequencies.tolist(),
                    'psd': psd.tolist(),
                    'band_powers': band_powers,
                    'spectral_features': spectral_features,
                    'dominant_frequency': dominant_frequency
                }
                
            except Exception as e:
                warnings.warn(f"Could not analyze frequency for layer {layer_name}: {e}")
                continue
                
        self.frequency_analysis = frequency_results
        return frequency_results
    
    def _extract_band_powers(self, frequencies: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Extract power in each frequency band"""
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency indices in this band
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            
            if np.any(band_mask):
                band_power = np.sum(psd[band_mask])
            else:
                band_power = 0.0
                
            band_powers[band_name] = band_power
            
        return band_powers
    
    def _compute_spectral_features(self, frequencies: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Compute additional spectral features"""
        return {
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd) if np.sum(psd) > 0 else 0,
            'spectral_entropy': self._spectral_entropy(psd),
            'peak_frequency': frequencies[np.argmax(psd)],
            'bandwidth': self._spectral_bandwidth(frequencies, psd)
        }
    
    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """Compute spectral entropy of the PSD."""
        if np.sum(psd) == 0:
            return 0.0
        normalized_psd = psd / np.sum(psd)
        # Avoid log(0) for zero probabilities
        normalized_psd = normalized_psd[normalized_psd > 0] 
        return -np.sum(normalized_psd * np.log2(normalized_psd))

    def _spectral_bandwidth(self, frequencies: np.ndarray, psd: np.ndarray) -> float:
        """Compute spectral bandwidth (e.g., 99% power bandwidth)."""
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0
        
        cumulative_power = np.cumsum(psd)
        # Find frequencies that enclose 99% of the power
        # This is a simplified approach, a more robust method might involve interpolation
        lower_bound_idx = np.where(cumulative_power >= 0.005 * total_power)[0][0] # 0.5% for lower
        upper_bound_idx = np.where(cumulative_power >= 0.995 * total_power)[0][0] # 99.5% for upper
        
        return frequencies[upper_bound_idx] - frequencies[lower_bound_idx]
    
    def classify_operational_states(self, frequency_analysis: Optional[Dict] = None) -> List[str]:
        """
        Classify the operational state of the neural network based on frequency patterns.
        
        This acts like an 'EEG reading' of the NN's computational activity.
        """
        if frequency_analysis is None:
            frequency_analysis = self.frequency_analysis
        
        if not frequency_analysis:
            return ["no_analysis_data"]
            
        classified_states = []
        
        for layer_name, result in frequency_analysis.items():
            band_powers = result.get('band_powers', {})
            state = self._determine_state_from_spectrum(band_powers)
            classified_states.append(f"{layer_name}: {state}")
            
        return classified_states

    def _determine_state_from_spectrum(self, band_powers: Dict[str, float]) -> str:
        """
        Determine the operational state from power distribution across frequency bands.
        Simplified for demonstration; a real system would use more complex ML.
        """
        total_power = sum(band_powers.values())
        if total_power == 0:
            return "idle"
            
        normalized_powers = {band: power / total_power for band, power in band_powers.items()}
        
        # Example classification rules
        if normalized_powers.get('gamma', 0) > 0.4: # High gamma suggests active processing
            return "active_computation"
        elif normalized_powers.get('beta', 0) > 0.3 and normalized_powers.get('gamma', 0) < 0.2: # Beta dominant
            return "inference_mode"
        elif normalized_powers.get('alpha', 0) > 0.5: # Alpha dominant
            return "idle"
        elif normalized_powers.get('delta', 0) > 0.4: # Delta dominant
            return "deep_processing"
        else:
            return "mixed_state"

    def _compute_state_confidence(self, band_powers: Dict[str, float]) -> float:
        """Compute a confidence score for the classified state.
        (Placeholder: Real confidence would involve statistical models.)
        """
        total_power = sum(band_powers.values())
        if total_power == 0:
            return 0.0
        
        # Example: confidence based on dominance of a single band
        max_power_ratio = max(band_powers.values()) / total_power if total_power > 0 else 0
        return max_power_ratio

    def generate_report(self) -> Dict:
        """
        Generate a comprehensive report of the NN-EEG analysis.
        """
        report = {
            'report_info': {
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': 'NN-EEG-v0.1.0',
                'model_summary': str(self.model)
            },
            'analysis_parameters': {
                'sample_rate': self.sample_rate,
                'window_size': self.window_size,
                'overlap_ratio': self.overlap_ratio,
                'frequency_bands': self.frequency_bands
            },
            'temporal_signals_summary': {
                'num_layers_analyzed': len(self.temporal_signals),
                'signal_length_per_layer': {name: len(sig) for name, sig in self.temporal_signals.items()}
            },
            'frequency_analysis_results': {},
            'operational_state_classification': self.classify_operational_states(self.frequency_analysis),
            'recommendations': [],
            'status': 'success'
        }
        
        for layer_name, freq_data in self.frequency_analysis.items():
            report['frequency_analysis_results'][layer_name] = {
                'dominant_frequency': freq_data['dominant_frequency'],
                'band_powers': freq_data['band_powers'],
                'spectral_features': freq_data['spectral_features'],
                'overall_state': self._determine_state_from_spectrum(freq_data['band_powers'])
            }
        
        # Determine overall status of the report
        if self.frequency_analysis:
            report['status'] = 'success'
        else:
            report['status'] = 'partial_success'

        # Add simple recommendations based on dominant states
        if any("active_computation" in state for state in report['operational_state_classification']):
            report['recommendations'].append("Consider optimizing layers identified with high active computation.")
        
        return report

    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize the frequency analysis results for each layer.
        """
        if not self.frequency_analysis:
            print("No frequency analysis results to visualize.")
            return
        
        num_layers = len(self.frequency_analysis)
        if num_layers == 0:
            print("No layers with frequency analysis results to visualize.")
            return
        
        cols = 2 # max 2 columns for now
        rows = (num_layers + cols - 1) // cols # calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (layer_name, result) in enumerate(self.frequency_analysis.items()):
            if i >= len(axes):
                break # Don't try to plot more than available subplots

            ax = axes[i]
            frequencies = np.array(result['frequencies'])
            psd = np.array(result['psd'])

            if len(frequencies) == 0 or len(psd) == 0:
                ax.set_title(f'{layer_name} (No data)')
                continue

            ax.plot(frequencies, psd, 'b-', linewidth=2, label='NN-EEG PSD')
            ax.set_title(f"{layer_name} - Dominant: {result['dominant_frequency']:.2f} Hz")
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 50) # Limit x-axis for better visibility of common EEG bands
            
            # Highlight frequency bands
            for band_name, (low, high) in self.frequency_bands.items():
                ax.axvspan(low, high, color='gray', alpha=0.1, label=f'{band_name} band')

        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        
        if save_path:
            # Ensure the directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def cleanup(self):
        """
        Clean up resources, primarily unregistering hooks.
        """
        self.activation_capture.cleanup()
        print("NN-EEG analyzer cleanup complete")


# Example usage (for quick testing/demonstration)
# These functions are typically called from an integration framework

def create_test_model() -> nn.Module:
    """Create a simple neural network for testing."""
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
    return model

def run_cifar10_demonstration():
    """
    Demonstrates NN-EEG analysis on a synthetic CIFAR-10 like dataset.
    """
    print("\nRunning NN-EEG CIFAR-10 Demonstration...")
    
    # Create a simple model
    model = create_test_model()
    
    # Create synthetic CIFAR-10 like data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Using a small synthetic dataset for quick demo
    synthetic_data = torch.randn(64, 3, 32, 32) # 64 samples, 3 channels, 32x32 image
    synthetic_targets = torch.randint(0, 10, (64,))
    dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Initialize and run NN-EEG analyzer
    eeg_analyzer = NeuralEEG(model)
    temporal_signals = eeg_analyzer.extract_temporal_signals(dataloader)
    frequency_results = eeg_analyzer.analyze_frequency_domain(temporal_signals)
    operational_state = eeg_analyzer.classify_operational_states(frequency_results)
    
    print(f"\nNN-EEG Operational State: {operational_state}")
    
    # Visualize results
    try:
        eeg_analyzer.visualize_results(save_path="results/nn_eeg_cifar10_demo.png")
        print("Visualization saved to results/nn_eeg_cifar10_demo.png")
    except Exception as e:
        print(f"Error saving visualization: {e}")

    # Generate and print report
    report = eeg_analyzer.generate_report()
    print("\nNN-EEG Analysis Report:")
    print(json.dumps(report, indent=2, default=str))
    
    eeg_analyzer.cleanup()

if __name__ == "__main__":
    run_cifar10_demonstration()