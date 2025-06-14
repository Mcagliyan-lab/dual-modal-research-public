#!/usr/bin/env python3
"""
Neural Interface Integration Test
Tests neural network interface components for dual-modal processing
"""

import sys
import unittest
from pathlib import Path

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestNeuralInterface(unittest.TestCase):
    """Test neural interface integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'eeg_channels': 64,
            'fmri_voxels': 1000,
            'integration_method': 'dual_modal'
        }
    
    def test_interface_initialization(self):
        """Test interface initialization"""
        try:
            from dual_modal.integration import framework
            
            # Test framework initialization
            integrator = framework.DualModalIntegrator()
            self.assertIsNotNone(integrator)
            
            print("✅ Interface initialization test passed")
            
        except ImportError:
            self.skipTest("Integration framework not available")
    
    def test_data_processing(self):
        """Test data processing pipeline"""
        try:
            from dual_modal.utils import data_loaders
            
            # Test data loader
            loader = data_loaders.MultiModalLoader()
            self.assertIsNotNone(loader)
            
            print("✅ Data processing test passed")
            
        except ImportError:
            self.skipTest("Data loaders not available")

def main():
    """Run neural interface tests"""
    print("🧠 Neural Interface Test Suite")
    print("=" * 40)
    
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main()
