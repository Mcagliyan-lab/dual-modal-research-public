import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
import traceback
from pathlib import Path
import sys

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import fixed framework
try:
    from src.nn_neuroimaging.integration.framework import DualModalIntegrator
    FRAMEWORK_AVAILABLE = True
    print("✅ DualModalIntegrator framework available.")
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    print(f"❌ DualModalIntegrator framework not available: {e}")
    sys.exit(1)

def safe_json_dumps(obj, indent=2):
    """Convert object to JSON string, handling numpy arrays safely."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_obj = convert_numpy(obj)
    return json.dumps(converted_obj, indent=indent)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def validate_on_real_cifar10():
    print("🚀 Starting Real Dataset Validation on CIFAR-10...")
    print("=" * 60)

    # 1. Load CIFAR-10 Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Download data to 'data' directory if not already present
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
        print("✅ CIFAR-10 training dataset loaded.")
    except Exception as e:
        print(f"❌ Failed to load CIFAR-10 dataset: {e}")
        traceback.print_exc()
        return

    # 2. Initialize a model (using SimpleCNN for demonstration)
    model = SimpleCNN()
    model.eval() # Set model to evaluation mode

    # 3. Perform analysis using DualModalIntegrator
    print("\n🔬 Performing Dual-Modal Analysis on CIFAR-10 data...")
    try:
        # Adjusted fmri_grid_size to match CIFAR-10 input (32x32)
        analyzer = DualModalIntegrator(model, grid_size=(32, 32, 3)) 
        
        # Analyze a subset of the data for quicker testing
        # The analyze method expects a dataloader.
        # It's important to specify a reasonable max_batches if the dataset is large.
        results = analyzer.analyze(trainloader, report_type='technical') # Removed max_batches to allow analyze to iterate over the entire dataloader
        print("✅ Dual-Modal Analysis completed for CIFAR-10.")
        
        print(f"DEBUG: Keys in results: {results.keys()}")
        if 'analysis_results' in results:
            print(f"DEBUG: Keys in results['analysis_results']: {results['analysis_results'].keys()}")
            if 'cross_modal_validation' in results['analysis_results']:
                print(f"DEBUG: Keys in results['analysis_results']['cross_modal_validation']: {results['analysis_results']['cross_modal_validation'].keys()}")
                
        # Save results
        output_dir = Path("results/real_dataset_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "cifar10_validation_results.json"
        
        with open(results_file, 'w') as f:
            f.write(safe_json_dumps(results))
        print(f"✅ Results saved to {results_file}")

        # Extract and print key consistency scores
        consistency_score = results.get('cross_modal_validation', {}).get('overall_consistency_score', 'N/A')
        print(f"🎯 Overall Cross-Modal Consistency Score on CIFAR-10: {consistency_score}")

    except Exception as e:
        print(f"❌ Dual-Modal Analysis failed: {e}")
        traceback.print_exc()
    finally:
        if 'analyzer' in locals() and analyzer:
            analyzer.cleanup()

    print("\n📈 Real Dataset Validation Finished.")
    print("=" * 60)

if __name__ == "__main__":
    validate_on_real_cifar10() 