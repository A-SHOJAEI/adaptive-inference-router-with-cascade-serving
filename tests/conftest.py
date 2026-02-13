"""Pytest configuration and fixtures for adaptive inference router tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_inference_router_with_cascade_serving.utils.config import Config


@pytest.fixture
def device():
    """Fixture providing device for testing."""
    return torch.device("cpu")  # Use CPU for testing


@pytest.fixture
def test_config() -> Config:
    """Fixture providing test configuration."""
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
        "training": {
            "epochs": 5,
            "patience": 3,
            "rl": {
                "learning_rate": 1e-3,
                "clip_epsilon": 0.2,
                "entropy_coeff": 0.01,
                "value_loss_coeff": 0.5,
                "gamma": 0.99,
                "lambda_gae": 0.95,
            },
            "objectives": {
                "latency_weight": 0.4,
                "accuracy_weight": 0.3,
                "throughput_weight": 0.2,
                "sla_weight": 0.1,
            },
            "optimizer": {
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "min_lr": 1e-6,
            }
        },
        "data": {
            "batch_size": 16,
            "num_workers": 0,  # No multiprocessing for tests
            "pin_memory": False,
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
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
        },
        "environment": {
            "sla": {
                "p95_latency_ms": 100,
                "min_accuracy": 0.95,
            },
            "serving": {},
            "resources": {},
        },
        "evaluation": {
            "target_metrics": {
                "p99_latency_reduction_vs_static": 0.35,
                "accuracy_degradation_vs_full_model": 0.02,
                "throughput_improvement_rps": 2.5,
                "sla_violation_rate": 0.01,
            },
            "eval_frequency": 2,
            "eval_batch_size": 32,
            "num_eval_episodes": 10,
            "statistical_significance_alpha": 0.05,
        },
        "experiment": {
            "name": "test_experiment",
            "mlflow": {
                "tracking_uri": "./test_mlruns",
                "experiment_name": "test_adaptive_router",
            },
            "logging": {
                "level": "INFO",
                "log_dir": "./test_logs",
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        },
        "system": {
            "device": "cpu",
            "seed": 42,
            "deterministic": True,
            "benchmark": False,
            "checkpoint": {
                "save_dir": "./test_checkpoints",
                "save_frequency": 2,
            },
            "monitoring": {
                "log_system_metrics": False,  # Disable for tests
            }
        }
    }

    return Config(**config_dict)


@pytest.fixture
def sample_features(device) -> torch.Tensor:
    """Fixture providing sample feature tensor."""
    batch_size = 8
    num_features = 14  # 5 + 4 + 5 features

    # Set seed for reproducibility
    torch.manual_seed(42)

    return torch.randn(batch_size, num_features, device=device)


@pytest.fixture
def sample_sla_constraints(device) -> torch.Tensor:
    """Fixture providing sample SLA constraints."""
    batch_size = 8

    # [target_latency_ms, accuracy_threshold]
    constraints = torch.tensor([
        [100.0, 0.95],
        [150.0, 0.90],
        [75.0, 0.97],
        [200.0, 0.85],
        [50.0, 0.99],
        [120.0, 0.92],
        [80.0, 0.96],
        [180.0, 0.88],
    ], device=device)

    return constraints


@pytest.fixture
def sample_optimal_routes(device) -> torch.Tensor:
    """Fixture providing sample optimal routing decisions."""
    # Corresponding to the SLA constraints above
    routes = torch.tensor([1, 2, 0, 3, 0, 1, 0, 3], device=device)
    return routes


@pytest.fixture
def sample_batch(sample_features, sample_sla_constraints, sample_optimal_routes) -> Dict[str, torch.Tensor]:
    """Fixture providing a complete sample batch."""
    batch_size = sample_features.shape[0]

    return {
        "features": sample_features,
        "sla_constraints": sample_sla_constraints,
        "optimal_route": sample_optimal_routes,
        "performance": torch.randn(batch_size, 4),  # Mock performance metrics
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture providing temporary directory."""
    return tmp_path


@pytest.fixture
def setup_seeds():
    """Fixture to setup reproducible random seeds."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    yield

    # Cleanup (optional)
    pass


@pytest.fixture(scope="session", autouse=True)
def suppress_mlflow_warnings():
    """Suppress MLflow warnings during testing."""
    import warnings
    import logging

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

    # Set MLflow logging level to ERROR
    logging.getLogger("mlflow").setLevel(logging.ERROR)


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration function."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Custom assertions
class CustomAssertions:
    """Custom assertion methods for tests."""

    @staticmethod
    def assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6):
        """Assert two tensors are close."""
        assert torch.allclose(actual, expected, atol=atol), \
            f"Tensors not close: {actual} vs {expected}"

    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: {tensor.shape} vs {expected_shape}"

    @staticmethod
    def assert_probability_distribution(probs: torch.Tensor, dim: int = -1):
        """Assert tensor represents valid probability distribution."""
        # Check non-negative
        assert torch.all(probs >= 0), "Probabilities must be non-negative"

        # Check sums to 1 (approximately)
        sums = torch.sum(probs, dim=dim)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Probabilities must sum to 1"

    @staticmethod
    def assert_performance_metrics_valid(metrics: Dict[str, float]):
        """Assert performance metrics are in valid ranges."""
        if "latency" in metrics:
            assert metrics["latency"] > 0, "Latency must be positive"

        if "accuracy" in metrics:
            assert 0 <= metrics["accuracy"] <= 1, "Accuracy must be in [0, 1]"

        if "throughput" in metrics:
            assert metrics["throughput"] > 0, "Throughput must be positive"


@pytest.fixture
def custom_assertions():
    """Fixture providing custom assertion methods."""
    return CustomAssertions()