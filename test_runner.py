#!/usr/bin/env python3
"""Simple test runner to check for basic import and functionality issues."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")

    try:
        from adaptive_inference_router_with_cascade_serving.utils.config import Config
        print("✓ Config import successful")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader, MLPerfInferenceDataset
        print("✓ Data loader imports successful")
    except Exception as e:
        print(f"✗ Data loader imports failed: {e}")
        traceback.print_exc()
        return False

    try:
        from adaptive_inference_router_with_cascade_serving.models.model import AdaptiveInferenceRouter
        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model imports failed: {e}")
        traceback.print_exc()
        return False

    try:
        from adaptive_inference_router_with_cascade_serving.training.trainer import MultiObjectiveTrainer
        print("✓ Training imports successful")
    except Exception as e:
        print(f"✗ Training imports failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_basic_functionality():
    """Test basic functionality without full test framework."""
    print("\nTesting basic functionality...")

    try:
        from adaptive_inference_router_with_cascade_serving.utils.config import Config

        # Create test config
        config_dict = {
            "model": {
                "router": {
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "use_attention": True,
                    "attention_heads": 4,
                },
                "cascade": {
                    "variants": [
                        {"name": "quantized_int8", "latency_multiplier": 0.3, "accuracy_multiplier": 0.95, "memory_multiplier": 0.25},
                        {"name": "pruned_50", "latency_multiplier": 0.5, "accuracy_multiplier": 0.97, "memory_multiplier": 0.5},
                        {"name": "distilled", "latency_multiplier": 0.4, "accuracy_multiplier": 0.94, "memory_multiplier": 0.3},
                        {"name": "full_precision", "latency_multiplier": 1.0, "accuracy_multiplier": 1.0, "memory_multiplier": 1.0},
                    ]
                }
            },
            "data": {
                "batch_size": 16,
                "difficulty_features": [
                    "input_complexity",
                    "computational_graph_depth",
                    "memory_footprint",
                    "numerical_stability",
                    "inference_uncertainty",
                ],
                "sla_features": [
                    "target_latency_ms",
                    "accuracy_threshold",
                    "priority_level",
                    "client_tier",
                ],
                "system_features": [
                    "cpu_utilization",
                    "gpu_utilization",
                    "memory_usage",
                    "network_bandwidth",
                    "queue_length",
                ],
            },
            "system": {
                "device": "cpu",
                "seed": 42,
                "monitoring": {
                    "log_system_metrics": False
                }
            },
            "environment": {
                "sla": {
                    "p95_latency_ms": 100,
                    "min_accuracy": 0.95,
                }
            }
        }

        config = Config(**config_dict)
        print("✓ Config creation successful")

        # Test model creation
        from adaptive_inference_router_with_cascade_serving.models.model import AdaptiveInferenceRouter
        import torch

        model = AdaptiveInferenceRouter(config)
        print("✓ Model creation successful")

        # Test forward pass
        batch_size = 4
        input_dim = len(config.data.difficulty_features) + len(config.data.sla_features) + len(config.data.system_features)
        features = torch.randn(batch_size, input_dim)

        predictions = model(features)
        print("✓ Model forward pass successful")

        # Test data loading
        from adaptive_inference_router_with_cascade_serving.data.loader import AdaptiveDataLoader

        data_loader = AdaptiveDataLoader(config)
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()
        print("✓ Dataset creation successful")

        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        print("✓ DataLoader creation successful")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running basic functionality tests...\n")

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    print(f"\n{'='*50}")
    if success:
        print("✓ All basic tests passed!")
    else:
        print("✗ Some tests failed!")
    print(f"{'='*50}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())