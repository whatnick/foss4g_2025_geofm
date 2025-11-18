#!/usr/bin/env python3
"""
TerraTorch Import Test

This script tests different import methods for TerraTorch and TerraMind
to help debug import issues.
"""

import sys

def test_terratorch_imports():
    """Test different TerraTorch import methods."""
    print("Testing TerraTorch import methods...\n")
    
    # Test 1: Basic TerraTorch import
    try:
        import terratorch
        print("✅ Basic terratorch import successful")
        print(f"   Version: {getattr(terratorch, '__version__', 'unknown')}")
        print(f"   Location: {terratorch.__file__}")
        
        # Check available attributes
        attrs = [attr for attr in dir(terratorch) if not attr.startswith('_')]
        print(f"   Available attributes: {attrs[:10]}...")  # Show first 10
        
    except ImportError as e:
        print(f"❌ Basic terratorch import failed: {e}")
        return
    
    # Test 2: Try build_model import
    try:
        from terratorch.models import build_model
        print("✅ terratorch.models.build_model import successful")
    except ImportError as e:
        print(f"❌ terratorch.models.build_model import failed: {e}")
    
    # Test 3: Try BACKBONE_REGISTRY import
    try:
        from terratorch.models.backbones import BACKBONE_REGISTRY
        print("✅ terratorch.models.backbones.BACKBONE_REGISTRY import successful")
    except ImportError as e:
        print(f"❌ terratorch.models.backbones.BACKBONE_REGISTRY import failed: {e}")
    
    # Test 4: Try alternative imports
    try:
        from terratorch import models
        print("✅ terratorch.models import successful")
        
        # Check what's available in models
        model_attrs = [attr for attr in dir(models) if not attr.startswith('_')]
        print(f"   Models attributes: {model_attrs}")
        
    except ImportError as e:
        print(f"❌ terratorch.models import failed: {e}")
    
    # Test 5: Check for model creation methods
    try:
        import terratorch
        
        model_methods = [
            'build_model', 'get_model', 'create_model', 'load_model',
            'build_backbone', 'get_backbone'
        ]
        
        print("\\nChecking for model creation methods:")
        for method in model_methods:
            if hasattr(terratorch, method):
                print(f"✅ terratorch.{method} available")
            else:
                print(f"❌ terratorch.{method} not available")
                
        # Check in submodules
        if hasattr(terratorch, 'models'):
            print("\\nChecking terratorch.models for methods:")
            for method in model_methods:
                if hasattr(terratorch.models, method):
                    print(f"✅ terratorch.models.{method} available")
                else:
                    print(f"❌ terratorch.models.{method} not available")
                    
    except Exception as e:
        print(f"Error checking model methods: {e}")

def test_pytorch():
    """Test PyTorch availability."""
    print("\\n" + "="*50)
    print("Testing PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")

if __name__ == "__main__":
    print("TerraTorch Import Diagnostics")
    print("=" * 50)
    
    test_pytorch()
    
    print("\\n" + "="*50)
    test_terratorch_imports()
    
    print("\\n" + "="*50)
    print("Diagnostic complete!")
    print("\\nIf you see import failures, try:")
    print("1. pip install --upgrade terratorch")
    print("2. pip install terratorch[all]")
    print("3. Check TerraTorch documentation for correct version")