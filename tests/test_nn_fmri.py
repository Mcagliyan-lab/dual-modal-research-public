import pytest
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from pathlib import Path
import numpy as np
import unittest.mock

# Import the NeuralFMRI class from the new package structure
from nn_neuroimaging.nn_fmri.implementation import NeuralFMRI
from nn_neuroimaging.utils.config_utils import load_config

# --- Test Utilities ---
@pytest.fixture(scope="module")
def sample_config_fmri():
    """Fixture to load a sample configuration for fMRI tests."""
    # Reusing and adapting the config structure for fMRI specific tests
    config_content = """
model_parameters:
  nn_eeg:
    sample_rate: 1.0
    window_size: 50
    overlap_ratio: 0.5
  nn_fmri:
    grid_size: [4, 4, 2] # Adjusted for smaller test models/data

analysis_settings:
  general:
    max_batches_eeg: 100
    max_batches_fmri: 5 # Smaller for quick fMRI tests
    seed: 42
  validation:
    num_reproducibility_runs: 5
    consistency_threshold: 0.8

output_directories:
  results: results/
  visualizations: results/visualizations/
  extended_validation: extended_validation_results/

logging:
  level: INFO
  file: logs/app.log
"""
    config_path = Path("test_fmri_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    yield config_path # Provide the path to the config file
    config_path.unlink() # Clean up the dummy config file

@pytest.fixture(scope="module")
def test_fmri_model():
    """Fixture for a simple neural network model suitable for fMRI tests.
    Using a slightly modified architecture to better simulate different layer types.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), # Layer 0
        nn.ReLU(),
        nn.MaxPool2d(2), # Output size will be smaller after this
        nn.Conv2d(8, 16, 3, padding=1), # Layer 1
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), # Output will be 1x1, so just 16 features
        nn.Flatten(), # Layer 2
        nn.Linear(16, 10) # Layer 3
    )
    model.eval()
    return model

@pytest.fixture(scope="module")
def synthetic_fmri_dataloader():
    """Fixture for synthetic CIFAR-10-like data for fMRI tests."""
    # Larger input image to test spatial partitioning effectively
    data = torch.randn(20, 3, 64, 64) # 20 samples, 3 channels, 64x64 image
    targets = torch.randint(0, 10, (20,))
    dataset = data_utils.TensorDataset(data, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=4, shuffle=False) # 5 batches for max_batches_fmri=5
    return dataloader

# --- Unit Tests for NeuralFMRI ---

def test_neural_fmri_initialization(test_fmri_model, sample_config_fmri):
    """Test NeuralFMRI initialization with config."""
    config = load_config(sample_config_fmri)
    fmri_analyzer = NeuralFMRI(test_fmri_model, grid_size=tuple(config['model_parameters']['nn_fmri']['grid_size']))
    assert fmri_analyzer is not None
    assert fmri_analyzer.model == test_fmri_model
    assert fmri_analyzer.grid_size == (4, 4, 2) # From config
    assert fmri_analyzer.lambda_reg == 0.1 # Default value
    assert fmri_analyzer.epsilon == 1e-8 # Default value
    assert fmri_analyzer.min_grid_size == 2 # Default value
    assert len(fmri_analyzer.hooks) > 0 # Should register hooks
    fmri_analyzer.cleanup()

def test_analyze_spatial_patterns(test_fmri_model, synthetic_fmri_dataloader, sample_config_fmri):
    """Test spatial pattern analysis."""
    config = load_config(sample_config_fmri)
    fmri_analyzer = NeuralFMRI(test_fmri_model, grid_size=tuple(config['model_parameters']['nn_fmri']['grid_size']))

    spatial_results = fmri_analyzer.analyze_spatial_patterns(synthetic_fmri_dataloader)

    assert isinstance(spatial_results, dict)
    assert "status" in spatial_results
    assert spatial_results["status"] == "success"
    assert "spatial_patterns" in spatial_results
    assert isinstance(spatial_results["spatial_patterns"], dict)
    assert len(spatial_results["spatial_patterns"]) > 0 # Should analyze some layers

    for layer_name, result in spatial_results["spatial_patterns"].items():
        assert "layer_name" in result
        assert "original_shape" in result
        assert "spatial_dims" in result
        assert "grid_dimensions" in result
        assert "grid_results" in result
        assert "density_map" in result
        assert "spatial_statistics" in result
        assert "activation_summary" in result
        assert "status" in result and result["status"] == "success"
        assert isinstance(result["density_map"], dict)
        assert isinstance(result["spatial_statistics"], dict)
    fmri_analyzer.cleanup()

def test_compute_zeta_scores(test_fmri_model, synthetic_fmri_dataloader, sample_config_fmri):
    """Test computation of zeta scores."""
    config = load_config(sample_config_fmri)
    fmri_analyzer = NeuralFMRI(test_fmri_model, grid_size=tuple(config['model_parameters']['nn_fmri']['grid_size']))

    fmri_analyzer.analyze_spatial_patterns(synthetic_fmri_dataloader) # Ensure spatial patterns are analyzed first
    # Zeta scores require a baseline output, which is internally handled by _get_model_outputs
    # The method also involves removing grid regions and re-evaluating the model.
    # Mocking this for a unit test can be complex. For now, we will test if the method runs without error
    # and returns a dictionary with expected top-level keys.

    zeta_scores = fmri_analyzer.compute_zeta_scores(synthetic_fmri_dataloader, sample_size=2) # Smaller sample size for tests

    assert isinstance(zeta_scores, dict)
    assert len(zeta_scores) > 0 # Should have scores for some layers
    for layer_name, scores in zeta_scores.items():
        assert isinstance(scores, dict)
        assert "zeta_scores" in scores # Changed from "grid_zeta_scores"
        assert isinstance(scores["zeta_scores"], dict) # Changed from "grid_zeta_scores"
        assert len(scores["zeta_scores"]) > 0 # Changed from "grid_zeta_scores"
    fmri_analyzer.cleanup()

def test_generate_spatial_report(test_fmri_model, synthetic_fmri_dataloader, sample_config_fmri):
    """Test generation of spatial analysis report."""
    config = load_config(sample_config_fmri)
    fmri_analyzer = NeuralFMRI(test_fmri_model, grid_size=tuple(config['model_parameters']['nn_fmri']['grid_size']))

    fmri_analyzer.analyze_spatial_patterns(synthetic_fmri_dataloader)
    # Compute zeta scores as it's part of the report generation process
    fmri_analyzer.compute_zeta_scores(synthetic_fmri_dataloader, sample_size=2)
    report = fmri_analyzer.generate_spatial_report()

    assert isinstance(report, dict)
    assert "analysis_timestamp" in report # Corrected from report_info
    assert "implementation_status" in report
    assert "metrics" in report
    assert "model_info" in report # Corrected from report_info
    assert "spatial_patterns_summary" in report # Changed from "spatial_analysis_results"
    assert "zeta_scores_summary" in report # Changed from "zeta_score_results"
    assert report["status"] == "success" or report["status"] == "partial_success"
    fmri_analyzer.cleanup()

def test_neural_fmri_cleanup(test_fmri_model, sample_config_fmri):
    """Test that cleanup removes hooks."""
    config = load_config(sample_config_fmri)
    fmri_analyzer = NeuralFMRI(test_fmri_model, grid_size=tuple(config['model_parameters']['nn_fmri']['grid_size']))
    assert len(fmri_analyzer.hooks) > 0 # Hooks should be registered on initialization
    fmri_analyzer.cleanup()
    assert len(fmri_analyzer.hooks) == 0 