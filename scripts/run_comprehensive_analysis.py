import torch
import torch.nn as nn
import torch.utils.data as data_utils
import sys
import io

# Add the project root to the sys.path to allow imports from nn_neuroimaging
sys.path.insert(0, "./src")

from nn_neuroimaging.integration.framework import DualModalIntegrator

# --- Test Utilities (replicated from test_nn_eeg.py and test_nn_fmri.py) ---
def get_test_model():
    """Provides a simple neural network model for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    model.eval()
    return model

def get_synthetic_dataloader():
    """Provides synthetic CIFAR-10-like data."""
    data = torch.randn(200, 3, 32, 32) # 200 samples for 25 batches (batch_size=8)
    targets = torch.randint(0, 10, (200,))
    dataset = data_utils.TensorDataset(data, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=8, shuffle=False)
    return dataloader

if __name__ == "__main__":
    # Redirect stdout to a file with UTF-8 encoding
    original_stdout = sys.stdout
    output_file_path = "comprehensive_analysis_output.txt"
    with open(output_file_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        
        try:
            print("Running Comprehensive Dual-Modal Analysis Example")
            print("=" * 60)
            
            model = get_test_model()
            dataloader = get_synthetic_dataloader()
            
            # Initialize the DualModalIntegrator
            # Use a grid_size that aligns with the model and expected fMRI analysis
            # (8,8,4) is a common default for the fMRI component.
            integrator = DualModalIntegrator(model, grid_size=(8, 8, 4))
            
            # Perform a comprehensive analysis
            print("\nStarting comprehensive analysis...")
            results = integrator.analyze(dataloader, analysis_type='comprehensive', report_type='technical')
            
            print("\nComprehensive Analysis Results Summary:")
            print(f"  Overall Status: {results.get('status', 'N/A')}")
            print(f"  Processing Time: {results.get('performance_metrics', {}).get('processing_time_seconds', 'N/A'):.2f}s")
            print(f"  Cross-Modal Consistency Score: {results.get('cross_modal_validation', {}).get('overall_consistency_score', 'N/A'):.2f}")
            print(f"  Validation Level: {results.get('cross_modal_validation', {}).get('validation_level', 'N/A')}")
            
            print("\nDetailed Cross-Modal Validation Results:")
            for key, value in results.get('cross_modal_validation', {}).items():
                if key != 'validation_details': # Avoid printing verbose details here
                    print(f"  {key}: {value}")
            
            # Optionally, print full report or specific sections
            # print("\nFull Technical Report:")
            # print(json.dumps(results.get('summary', {}), indent=2))
            
        except Exception as e:
            print(f"An error occurred during comprehensive analysis: {e}")
        finally:
            # Ensure cleanup is called even if an error occurs
            if 'integrator' in locals() and integrator is not None:
                integrator.cleanup()
                print("\nCleanup complete.")
            
            print("=" * 60)
            print("Comprehensive Dual-Modal Analysis Example Finished.")
            
    sys.stdout = original_stdout # Restore original stdout 