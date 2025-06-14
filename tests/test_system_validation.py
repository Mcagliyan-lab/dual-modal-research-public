#!/usr/bin/env python3
"""
System Validation Test
Basic system validation and dependency checks
"""

import sys
import unittest
from pathlib import Path

class TestSystemValidation(unittest.TestCase):
    """Test system validation"""
    
    def test_dependencies(self):
        """Test required dependencies"""
        try:
            import numpy as np
            import torch
            
            self.assertTrue(hasattr(np, 'array'))
            self.assertTrue(hasattr(torch, 'tensor'))
            
            print("✅ Dependencies validated")
            
        except ImportError as e:
            self.fail(f"Dependency check failed: {e}")
    
    def test_path_configuration(self):
        """Test path configuration"""
        src_path = Path(__file__).parent.parent / 'src'
        self.assertTrue(src_path.exists())
        
        print("✅ Path configuration validated")

def main():
    """Run system validation tests"""
    print("🔧 System Validation Test Suite")
    print("=" * 40)
    
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main()
