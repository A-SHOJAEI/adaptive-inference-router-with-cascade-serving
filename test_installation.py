#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality.
This script should be run after installing dependencies.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")

    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

    try:
        # Test core module imports
        print("  - Testing config module...")
        from adaptive_inference_router_with_cascade_serving.utils.config import Config
        print("  ✓ Config imported successfully")

        print("  - Testing model module...")
        from adaptive_inference_router_with_cascade_serving.models.model import AdaptiveInferenceRouter
        print("  ✓ AdaptiveInferenceRouter imported successfully")

        print("  - Testing data loader...")
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader
        print("  ✓ AdaptiveDataLoader imported successfully")

        print("  - Testing trainer...")
        from adaptive_inference_router_with_cascade_serving.training.trainer import MultiObjectiveTrainer
        print("  ✓ MultiObjectiveTrainer imported successfully")

        print("  - Testing metrics...")
        from adaptive_inference_router_with_cascade_serving.evaluation.metrics import RoutingMetrics
        print("  ✓ RoutingMetrics imported successfully")

        return True

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("\nTesting GPU availability...")

    try:
        import torch

        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  - GPU not available, will use CPU")

        return True

    except ImportError:
        print("  ✗ PyTorch not installed")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    try:
        from adaptive_inference_router_with_cascade_serving.utils.config import load_config, get_default_config_path

        config_path = get_default_config_path()
        print(f"  - Default config path: {config_path}")

        if config_path.exists():
            config = load_config(config_path)
            print("  ✓ Configuration loaded successfully")
            print(f"  - Model hidden dim: {config.model.router.get('hidden_dim', 'Not set')}")
            print(f"  - Training epochs: {config.training.epochs}")
            print(f"  - Device: {config.system.device}")
            return True
        else:
            print(f"  ✗ Config file not found: {config_path}")
            return False

    except Exception as e:
        print(f"  ✗ Config loading error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")

    try:
        from adaptive_inference_router_with_cascade_serving.utils.config import Config
        from adaptive_inference_router_with_cascade_serving.models.model import AdaptiveInferenceRouter
        import torch

        # Create a minimal config
        config = Config()

        # Create model
        model = AdaptiveInferenceRouter(config)
        print("  ✓ Model created successfully")

        # Test forward pass
        batch_size = 4
        input_dim = model.input_dim
        features = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = model(features)

        print(f"  ✓ Forward pass successful, output keys: {list(output.keys())}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True

    except Exception as e:
        print(f"  ✗ Model creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Adaptive Inference Router - Installation Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_gpu_availability,
        test_config_loading,
        test_model_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation appears to be working correctly.")
        print("\nYou can now run training with:")
        print("  python scripts/train.py --help")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())