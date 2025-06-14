"""
NN-fMRI Spatial Analysis Module

Provides spatial analysis capabilities for neural network interpretability
using fMRI-inspired 3D grid-based activation mapping.

Key Features:
- Grid-based spatial activation mapping
- Critical region identification
- 3D visualization support
- Multi-resolution analysis
"""

from .implementation import NeuralFMRI as NNfMRIAnalyzer

__all__ = [
    "NNfMRIAnalyzer"
]

# Module configuration
DEFAULT_CONFIG = {
    "grid_resolution": (64, 64, 64),
    "temporal_resolution": 2.0,
    "smoothing_kernel": 6.0,
    "threshold_percentile": 95,
    "activation_threshold": 0.5
}

def get_default_config():
    """Get default NN-fMRI configuration."""
    return DEFAULT_CONFIG.copy() 