#!/usr/bin/env python3
"""
Test suite for dual_modal package core functionality.

This module tests the basic functionality of the dual-modal neuroimaging framework
including imports, configuration, and basic API operations.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dual_modal


class TestDualModalCore:
    """Test core dual_modal package functionality."""
    
    def test_package_import(self) -> None:
        """Test basic package import functionality."""
        assert dual_modal.__version__ == "1.0.0"
        assert dual_modal.__author__ == "Research Team"
        assert dual_modal.__license__ == "MIT"
    
    def test_version_function(self) -> None:
        """Test get_version() function."""
        version = dual_modal.get_version()
        assert isinstance(version, str)
        assert version == "1.0.0"
    
    def test_config_function(self) -> None:
        """Test get_config() function."""
        config = dual_modal.get_config()
        assert isinstance(config, dict)
        
        # Check required sections
        assert "nn_eeg" in config
        assert "nn_fmri" in config
        assert "integration" in config
        
        # Check EEG configuration
        eeg_config = config["nn_eeg"]
        assert "temporal_window" in eeg_config
        assert "frequency_bands" in eeg_config
        assert "sampling_rate" in eeg_config
        assert eeg_config["sampling_rate"] == 250
        
        # Check frequency bands
        freq_bands = eeg_config["frequency_bands"]
        assert "delta" in freq_bands
        assert "theta" in freq_bands
        assert "alpha" in freq_bands
        assert "beta" in freq_bands
        assert "gamma" in freq_bands
    
    def test_status_function(self) -> None:
        """Test get_status() function."""
        status = dual_modal.get_status()
        assert isinstance(status, dict)
        
        # Check required fields
        assert "development_phase" in status
        assert "core_implementation" in status
        assert "validation_status" in status
        
        assert status["development_phase"] == "Academic Publication Preparation"
        assert status["core_implementation"] == "Complete"
    
    def test_validation_results_function(self) -> None:
        """Test get_validation_results() function."""
        results = dual_modal.get_validation_results()
        assert isinstance(results, dict)
        
        # Check CIFAR-10 results
        assert "cifar10" in results
        cifar10_results = results["cifar10"]
        
        assert "cross_modal_consistency" in cifar10_results
        assert "std_deviation" in cifar10_results
        assert "processing_time" in cifar10_results
        assert "dataset_size" in cifar10_results
        
        # Validate key metrics
        consistency = cifar10_results["cross_modal_consistency"]
        assert isinstance(consistency, float)
        assert 0.9 <= consistency <= 1.0  # Should be around 89.7%
    
    def test_core_classes_available(self) -> None:
        """Test that core classes are importable."""
        assert hasattr(dual_modal, "DualModalFramework")
        assert hasattr(dual_modal, "NNEEGAnalyzer")
        assert hasattr(dual_modal, "NNfMRIAnalyzer")
        
        # Check if classes are actually classes/types
        assert callable(dual_modal.DualModalFramework)
        assert callable(dual_modal.NNEEGAnalyzer)
        assert callable(dual_modal.NNfMRIAnalyzer)
    
    def test_utility_functions_available(self) -> None:
        """Test that utility functions are importable."""
        assert hasattr(dual_modal, "load_config")
        assert callable(dual_modal.load_config)
    
    def test_config_immutability(self) -> None:
        """Test that config returns a copy (immutable)."""
        config1 = dual_modal.get_config()
        config2 = dual_modal.get_config()
        
        # Modify one config
        config1["nn_eeg"]["sampling_rate"] = 500
        
        # Check that the other is unchanged
        assert config2["nn_eeg"]["sampling_rate"] == 250
        assert config1 is not config2
    
    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        all_exports = dual_modal.__all__
        
        # Core classes should be exported
        assert "DualModalFramework" in all_exports
        assert "NNEEGAnalyzer" in all_exports
        assert "NNfMRIAnalyzer" in all_exports
        
        # Core functions should be exported
        assert "load_config" in all_exports
        
        # Metadata should be exported
        assert "__version__" in all_exports
        assert "__author__" in all_exports
        assert "__license__" in all_exports


class TestDualModalConfiguration:
    """Test configuration validation and structure."""
    
    def test_default_config_structure(self) -> None:
        """Test the structure of default configuration."""
        config = dual_modal.get_config()
        
        # Test NN-EEG configuration
        eeg_config = config["nn_eeg"]
        assert isinstance(eeg_config["temporal_window"], int)
        assert isinstance(eeg_config["sampling_rate"], int)
        assert isinstance(eeg_config["frequency_bands"], dict)
        
        # Test frequency bands format
        for band_name, band_range in eeg_config["frequency_bands"].items():
            assert isinstance(band_name, str)
            assert isinstance(band_range, tuple)
            assert len(band_range) == 2
            assert band_range[0] < band_range[1]  # Low < High frequency
        
        # Test NN-fMRI configuration
        fmri_config = config["nn_fmri"]
        assert isinstance(fmri_config["grid_resolution"], tuple)
        assert len(fmri_config["grid_resolution"]) == 3  # 3D grid
        assert isinstance(fmri_config["temporal_resolution"], float)
        assert isinstance(fmri_config["smoothing_kernel"], float)
        
        # Test integration configuration
        integration_config = config["integration"]
        assert isinstance(integration_config["consistency_threshold"], float)
        assert isinstance(integration_config["weight_temporal"], float)
        assert isinstance(integration_config["weight_spatial"], float)
        
        # Weights should sum to 1.0 (approximately)
        total_weight = (integration_config["weight_temporal"] + 
                       integration_config["weight_spatial"])
        assert abs(total_weight - 1.0) < 0.01


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 
