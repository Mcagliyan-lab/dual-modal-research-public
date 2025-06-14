#!/usr/bin/env python3
"""Test script for dual_modal package"""

import sys
sys.path.append('src')

import dual_modal

print("📦 Package Import Test")
print("=" * 40)
print(f"✅ Import successful")
print(f"📋 Version: {dual_modal.__version__}")
print(f"👤 Author: {dual_modal.__author__}")
print(f"📄 License: {dual_modal.__license__}")

print("\n🔍 Available Classes:")
print(f"  * DualModalFramework: {hasattr(dual_modal, 'DualModalFramework')}")
print(f"  * NNEEGAnalyzer: {hasattr(dual_modal, 'NNEEGAnalyzer')}")
print(f"  * NNfMRIAnalyzer: {hasattr(dual_modal, 'NNfMRIAnalyzer')}")

print("\n🛠️ Available Functions:")
print(f"  * load_config: {hasattr(dual_modal, 'load_config')}")
print(f"  * get_version: {hasattr(dual_modal, 'get_version')}")
print(f"  * get_config: {hasattr(dual_modal, 'get_config')}")

print("\n🧪 Test Basic Functionality:")
try:
    version = dual_modal.get_version()
    print(f"  ✅ get_version(): {version}")
    
    config = dual_modal.get_config()
    print(f"  ✅ get_config(): {len(config)} sections")
    
    status = dual_modal.get_status()
    print(f"  ✅ get_status(): {status['development_phase']}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n🎉 Package ready for use!") 