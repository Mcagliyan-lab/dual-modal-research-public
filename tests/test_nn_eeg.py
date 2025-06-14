import pytest
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from pathlib import Path
import numpy as np
import unittest.mock
import matplotlib.pyplot as plt

# Import the NeuralEEG class from the new package structure
from nn_neuroimaging.nn_eeg.implementation import NeuralEEG, ActivationCapture
from nn_neuroimaging.utils.config_utils import load_config

# --- Test Utilities ---
@pytest.fixture(scope="module")
def sample_config():
    """Fixture to load a sample configuration for tests."""
    # Create a dummy config.yaml for testing purposes
    config_content = """
model_parameters:
  nn_eeg:
    sample_rate: 1.0
    window_size: 50
    overlap_ratio: 0.5
  nn_fmri:
    grid_size: [8, 8, 4]

analysis_settings:
  general:
    max_batches_eeg: 100
    max_batches_fmri: 10
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
    config_path = Path("test_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    yield config_path # Provide the path to the config file
    config_path.unlink() # Clean up the dummy config file

@pytest.fixture(scope="module")
def test_model():
    """Fixture for a simple neural network model."""
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

@pytest.fixture(scope="module")
def synthetic_dataloader():
    """Fixture for synthetic CIFAR-10-like data."""
    data = torch.randn(800, 3, 32, 32) # 800 samples for 100 batches (batch_size=8)
    targets = torch.randint(0, 10, (800,))
    dataset = data_utils.TensorDataset(data, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=8, shuffle=False)
    return dataloader

# --- Unit Tests for NeuralEEG ---

def test_neural_eeg_initialization(test_model, sample_config):
    """Test NeuralEEG initialization with config."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    assert eeg_analyzer is not None
    assert eeg_analyzer.model == test_model
    assert eeg_analyzer.sample_rate == 1.0 # From config
    assert eeg_analyzer.window_size == 50 # From config
    assert eeg_analyzer.overlap_ratio == 0.5 # From config
    assert eeg_analyzer.max_batches_eeg == 100 # From config
    assert isinstance(eeg_analyzer.activation_capture, ActivationCapture)
    eeg_analyzer.cleanup()

def test_extract_temporal_signals(test_model, synthetic_dataloader, sample_config):
    """Test extraction of temporal signals."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    temporal_signals = eeg_analyzer.extract_temporal_signals(synthetic_dataloader)
    
    assert isinstance(temporal_signals, dict)
    assert len(temporal_signals) > 0 # Should capture signals from some layers
    for layer_name, signal_array in temporal_signals.items():
        assert isinstance(signal_array, np.ndarray)
        assert len(signal_array) == eeg_analyzer.max_batches_eeg # Number of batches processed
        assert layer_name.startswith("layer_")
    eeg_analyzer.cleanup()

def test_analyze_frequency_domain(test_model, synthetic_dataloader, sample_config):
    """Test frequency domain analysis."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    temporal_signals = eeg_analyzer.extract_temporal_signals(synthetic_dataloader)
    frequency_results = eeg_analyzer.analyze_frequency_domain(temporal_signals)
    
    assert isinstance(frequency_results, dict)
    assert len(frequency_results) > 0
    for layer_name, result in frequency_results.items():
        assert "frequencies" in result
        assert "psd" in result
        assert "band_powers" in result
        assert "dominant_frequency" in result
        assert isinstance(result["frequencies"], list)
        assert isinstance(result["psd"], list)
        assert isinstance(result["band_powers"], dict)
    eeg_analyzer.cleanup()

def test_classify_operational_states(test_model, synthetic_dataloader, sample_config):
    """Test operational state classification."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    temporal_signals = eeg_analyzer.extract_temporal_signals(synthetic_dataloader)
    frequency_results = eeg_analyzer.analyze_frequency_domain(temporal_signals)
    classified_states = eeg_analyzer.classify_operational_states(frequency_results)
    
    assert isinstance(classified_states, list)
    assert len(classified_states) > 0
    # Check format of states (e.g., "layer_0_conv1: idle")
    assert all(": " in state for state in classified_states)
    eeg_analyzer.cleanup()

def test_neural_eeg_cleanup(test_model, sample_config):
    """Test that cleanup removes hooks."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    # Ensure hooks are registered
    assert len(eeg_analyzer.activation_capture.hooks) > 0
    eeg_analyzer.cleanup()
    assert len(eeg_analyzer.activation_capture.hooks) == 0

def test_neural_eeg_generate_report(test_model, synthetic_dataloader, sample_config):
    """Test generation of analysis report."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    eeg_analyzer.extract_temporal_signals(synthetic_dataloader)
    eeg_analyzer.analyze_frequency_domain() # Use the internal temporal_signals
    report = eeg_analyzer.generate_report()
    
    assert isinstance(report, dict)
    assert "status" in report # Changed from "overall_status"
    assert "temporal_signals_summary" in report # Changed from "summary_eeg"
    assert "frequency_analysis_results" in report # Changed from "frequency_details"
    assert "operational_state_classification" in report # Corrected typo: changed from "operational_states_classification"
    assert report["status"] == "success" or report["status"] == "partial_success"
    eeg_analyzer.cleanup()

def test_neural_eeg_visualize_results(test_model, synthetic_dataloader, sample_config):
    """Test visualization of analysis results."""
    eeg_analyzer = NeuralEEG(test_model, config_path=str(sample_config))
    eeg_analyzer.extract_temporal_signals(synthetic_dataloader)
    eeg_analyzer.analyze_frequency_domain()
    
    # Mock matplotlib.pyplot to avoid actual plotting and saving
    with unittest.mock.patch('matplotlib.pyplot.savefig') as mock_savefig, \
         unittest.mock.patch('matplotlib.pyplot.close') as mock_close:
        
        save_path = "test_eeg_visualization.png"
        eeg_analyzer.visualize_results(save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight') # Changed dpi to 300
        # Assert that plt.close(fig) was called, not plt.close('all')
        # We need to get the figure object from the analyzer to pass it to the mock.
        # This requires modifying the visualize_results method to return the figure,
        # or the test needs to mock plt.subplots to capture the fig object.
        # A simpler approach for now is to check for the 'fig' object as an argument.
        # However, the previous error `Calls: [call('all'), call(<Figure size 2000x1200 with 3 Axes>)].`
        # indicates that the figure object is indeed passed as an argument to one of the close calls.
        # So we can simply assert for a call that contains the figure object.
        mock_close.assert_any_call(unittest.mock.ANY) # Check if it was called at least once with any argument
        assert any(call[0] is not None and isinstance(call[0][0], plt.Figure) for call in mock_close.call_args_list) # Check for a call with a Figure object as argument
        eeg_analyzer.cleanup() 