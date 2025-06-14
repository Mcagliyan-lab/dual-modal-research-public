"""
NN-EEG Temporal Analysis Module

Provides temporal analysis capabilities for neural network interpretability
using EEG-inspired frequency domain processing.

Key Features:
- FFT-based frequency domain analysis
- Multi-band temporal pattern extraction
- Real-time processing capabilities
- Configurable temporal windows
"""

from .implementation import NeuralEEG as NNEEGAnalyzer

__all__ = [
    "NNEEGAnalyzer"
]

# Module configuration
DEFAULT_CONFIG = {
    "temporal_window": 100,
    "frequency_bands": {
        "delta": (0.5, 4),
        "theta": (4, 8), 
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    },
    "sampling_rate": 250,
    "overlap": 0.5
}

def get_default_config():
    """Get default NN-EEG configuration."""
    return DEFAULT_CONFIG.copy() 