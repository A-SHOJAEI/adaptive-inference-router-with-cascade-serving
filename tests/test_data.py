"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from adaptive_inference_router_with_cascade_serving.data.loader import (
    MLPerfInferenceDataset,
    AdaptiveDataLoader
)
from adaptive_inference_router_with_cascade_serving.data.preprocessing import (
    QueryDifficultyEstimator,
    SLAConstraintProcessor,
    SystemLoadMonitor
)


class TestMLPerfInferenceDataset:
    """Test cases for MLPerfInferenceDataset."""

    @pytest.mark.unit
    def test_dataset_initialization(self, test_config, setup_seeds):
        """Test dataset initialization."""
        dataset = MLPerfInferenceDataset(
            size=100,
            seed=test_config.system.seed
        )

        assert len(dataset) == 100
        assert hasattr(dataset, 'features')
        assert hasattr(dataset, 'optimal_routes')
        assert hasattr(dataset, 'performance_metrics')

    @pytest.mark.unit
    def test_dataset_getitem(self, test_config, setup_seeds):
        """Test dataset __getitem__ method."""
        dataset = MLPerfInferenceDataset(size=50, seed=42)

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert 'features' in sample
        assert 'optimal_route' in sample
        assert 'performance' in sample
        assert 'sla_constraints' in sample

        # Check tensor types and shapes
        assert isinstance(sample['features'], torch.Tensor)
        assert isinstance(sample['optimal_route'], torch.Tensor)
        assert isinstance(sample['performance'], torch.Tensor)
        assert isinstance(sample['sla_constraints'], torch.Tensor)

        # Check feature dimension
        expected_features = len(dataset.feature_names)
        assert sample['features'].shape == (expected_features,)

    @pytest.mark.unit
    def test_dataset_statistics(self, setup_seeds):
        """Test dataset statistics computation."""
        dataset = MLPerfInferenceDataset(size=200, seed=42)
        stats = dataset.get_statistics()

        assert 'size' in stats
        assert stats['size'] == 200
        assert 'num_features' in stats
        assert 'route_distribution' in stats

        # Check that we have statistics for all features
        for feature_name in dataset.feature_names:
            assert f"{feature_name}_mean" in stats
            assert f"{feature_name}_std" in stats
            assert f"{feature_name}_min" in stats
            assert f"{feature_name}_max" in stats

    @pytest.mark.unit
    def test_dataset_reproducibility(self):
        """Test that dataset generation is reproducible."""
        dataset1 = MLPerfInferenceDataset(size=50, seed=42)
        dataset2 = MLPerfInferenceDataset(size=50, seed=42)

        # Check that the same seed produces the same data
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1['features'], sample2['features'])
        assert torch.equal(sample1['optimal_route'], sample2['optimal_route'])

    @pytest.mark.unit
    def test_custom_feature_sets(self, setup_seeds):
        """Test dataset with custom feature sets."""
        custom_difficulty = ["feature1", "feature2"]
        custom_sla = ["sla1"]
        custom_system = ["sys1", "sys2"]

        dataset = MLPerfInferenceDataset(
            size=30,
            difficulty_features=custom_difficulty,
            sla_features=custom_sla,
            system_features=custom_system,
            seed=42
        )

        sample = dataset[0]
        expected_dim = len(custom_difficulty + custom_sla + custom_system)
        assert sample['features'].shape == (expected_dim,)


class TestAdaptiveDataLoader:
    """Test cases for AdaptiveDataLoader."""

    @pytest.mark.unit
    def test_data_loader_initialization(self, test_config):
        """Test data loader initialization."""
        data_loader = AdaptiveDataLoader(test_config)

        assert data_loader.config == test_config
        assert hasattr(data_loader, 'scalers')
        assert hasattr(data_loader, 'encoders')

    @pytest.mark.unit
    def test_create_datasets(self, test_config, setup_seeds):
        """Test dataset creation."""
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

        # Check that datasets are created
        assert train_dataset is not None
        assert val_dataset is not None
        assert test_dataset is not None

        # Check dataset sizes
        total_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total_size == 50000  # Default size

    @pytest.mark.unit
    def test_create_dataloaders(self, test_config, setup_seeds):
        """Test data loader creation."""
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )

        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch sizes
        batch = next(iter(train_loader))
        assert batch['features'].shape[0] == test_config.data.batch_size

    @pytest.mark.unit
    def test_setup_preprocessing(self, test_config, setup_seeds):
        """Test preprocessing setup."""
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, _, _ = data_loader.create_datasets()

        data_loader.setup_preprocessing(train_dataset)

        # Check that scalers are fitted
        assert 'features' in data_loader.scalers

    @pytest.mark.unit
    def test_preprocess_batch(self, test_config, sample_batch, setup_seeds):
        """Test batch preprocessing."""
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, _, _ = data_loader.create_datasets()
        data_loader.setup_preprocessing(train_dataset)

        preprocessed_batch = data_loader.preprocess_batch(sample_batch)

        # Check that preprocessing was applied
        assert 'features' in preprocessed_batch
        assert preprocessed_batch['features'].shape == sample_batch['features'].shape

    @pytest.mark.unit
    def test_save_load_preprocessing_state(self, test_config, temp_dir, setup_seeds):
        """Test saving and loading preprocessing state."""
        data_loader = AdaptiveDataLoader(test_config)
        train_dataset, _, _ = data_loader.create_datasets()
        data_loader.setup_preprocessing(train_dataset)

        # Save preprocessing state
        save_path = temp_dir / "preprocessing_state.pkl"
        data_loader.save_preprocessing_state(save_path)

        # Create new loader and load state
        new_loader = AdaptiveDataLoader(test_config)
        new_loader.load_preprocessing_state(save_path)

        # Check that state was loaded
        assert 'features' in new_loader.scalers


class TestQueryDifficultyEstimator:
    """Test cases for QueryDifficultyEstimator."""

    @pytest.mark.unit
    def test_estimator_initialization(self, test_config):
        """Test estimator initialization."""
        estimator = QueryDifficultyEstimator(test_config)

        assert estimator.config == test_config
        assert not estimator.is_fitted
        assert estimator.feature_weights is None

    @pytest.mark.unit
    def test_estimator_fit(self, test_config, setup_seeds):
        """Test estimator fitting."""
        estimator = QueryDifficultyEstimator(test_config)

        # Create mock training data
        features = np.random.randn(100, 5)
        latencies = np.random.uniform(50, 200, 100)

        estimator.fit(features, latencies)

        assert estimator.is_fitted
        assert estimator.feature_weights is not None
        assert len(estimator.feature_weights) == 5

    @pytest.mark.unit
    def test_estimate_difficulty_fitted(self, test_config, setup_seeds):
        """Test difficulty estimation when fitted."""
        estimator = QueryDifficultyEstimator(test_config)

        # Fit estimator
        features = np.random.randn(100, 5)
        latencies = np.random.uniform(50, 200, 100)
        estimator.fit(features, latencies)

        # Test estimation
        test_features = np.random.randn(10, 5)
        difficulties = estimator.estimate_difficulty(test_features)

        assert len(difficulties) == 10
        assert np.all(difficulties >= 0)
        assert np.all(difficulties <= 1)

    @pytest.mark.unit
    def test_estimate_difficulty_unfitted(self, test_config, setup_seeds):
        """Test difficulty estimation when not fitted (heuristic mode)."""
        estimator = QueryDifficultyEstimator(test_config)

        # Test estimation without fitting
        test_features = np.random.randn(10, 5)
        difficulties = estimator.estimate_difficulty(test_features)

        assert len(difficulties) == 10
        assert np.all(difficulties >= 0)
        assert np.all(difficulties <= 1)

    @pytest.mark.unit
    def test_get_feature_importance(self, test_config, setup_seeds):
        """Test feature importance extraction."""
        estimator = QueryDifficultyEstimator(test_config)

        # Before fitting
        importance = estimator.get_feature_importance()
        assert importance is None

        # After fitting
        features = np.random.randn(100, 5)
        latencies = np.random.uniform(50, 200, 100)
        estimator.fit(features, latencies)

        importance = estimator.get_feature_importance()
        assert importance is not None
        assert len(importance) == 5

    @pytest.mark.unit
    def test_fit_validation(self, test_config):
        """Test fit method input validation."""
        estimator = QueryDifficultyEstimator(test_config)

        # Test mismatched shapes
        features = np.random.randn(100, 5)
        latencies = np.random.uniform(50, 200, 90)  # Wrong size

        with pytest.raises(ValueError):
            estimator.fit(features, latencies)


class TestSLAConstraintProcessor:
    """Test cases for SLAConstraintProcessor."""

    @pytest.mark.unit
    def test_processor_initialization(self, test_config):
        """Test processor initialization."""
        processor = SLAConstraintProcessor(test_config)

        assert processor.config == test_config
        assert hasattr(processor, 'constraint_validators')

    @pytest.mark.unit
    def test_process_valid_constraints(self, test_config):
        """Test processing valid constraints."""
        processor = SLAConstraintProcessor(test_config)

        raw_constraints = {
            "target_latency_ms": 100.0,
            "accuracy_threshold": 0.95,
            "priority_level": 2,
            "client_tier": 1,
        }

        processed = processor.process_constraints(raw_constraints)

        assert "target_latency_ms" in processed
        assert "accuracy_threshold" in processed
        assert "priority_level" in processed
        assert "client_tier" in processed

        # Check value ranges
        assert processed["target_latency_ms"] == 100.0
        assert processed["accuracy_threshold"] == 0.95
        assert 0 <= processed["priority_level"] <= 1
        assert 0 <= processed["client_tier"] <= 1

    @pytest.mark.unit
    def test_validate_latency(self, test_config):
        """Test latency validation."""
        processor = SLAConstraintProcessor(test_config)

        # Valid latency
        result = processor._validate_latency(100.0)
        assert result == 100.0

        # Invalid latencies
        with pytest.raises(ValueError):
            processor._validate_latency(-10.0)

        with pytest.raises(ValueError):
            processor._validate_latency("invalid")

    @pytest.mark.unit
    def test_validate_accuracy(self, test_config):
        """Test accuracy validation."""
        processor = SLAConstraintProcessor(test_config)

        # Valid accuracy
        result = processor._validate_accuracy(0.95)
        assert result == 0.95

        # Invalid accuracies
        with pytest.raises(ValueError):
            processor._validate_accuracy(1.5)

        with pytest.raises(ValueError):
            processor._validate_accuracy(-0.1)

    @pytest.mark.unit
    def test_validate_priority(self, test_config):
        """Test priority validation."""
        processor = SLAConstraintProcessor(test_config)

        # Valid priorities
        assert processor._validate_priority(0) == 0.0
        assert processor._validate_priority(3) == 1.0

        # Invalid priorities
        with pytest.raises(ValueError):
            processor._validate_priority(4)

        with pytest.raises(ValueError):
            processor._validate_priority(-1)

    @pytest.mark.unit
    def test_check_sla_violation(self, test_config):
        """Test SLA violation checking."""
        processor = SLAConstraintProcessor(test_config)

        # No violation
        performance = {"latency_ms": 80.0, "accuracy": 0.96}
        constraints = {"target_latency_ms": 100.0, "accuracy_threshold": 0.95}

        is_violation, violations = processor.check_sla_violation(performance, constraints)
        assert not is_violation
        assert len(violations) == 0

        # Latency violation
        performance = {"latency_ms": 120.0, "accuracy": 0.96}

        is_violation, violations = processor.check_sla_violation(performance, constraints)
        assert is_violation
        assert "latency" in violations

        # Accuracy violation
        performance = {"latency_ms": 80.0, "accuracy": 0.90}

        is_violation, violations = processor.check_sla_violation(performance, constraints)
        assert is_violation
        assert "accuracy" in violations

    @pytest.mark.unit
    def test_add_default_constraints(self, test_config):
        """Test default constraint addition."""
        processor = SLAConstraintProcessor(test_config)

        # Partial constraints
        partial_constraints = {"target_latency_ms": 150.0}

        processed = processor.process_constraints(partial_constraints)

        # Check that defaults were added
        assert "accuracy_threshold" in processed
        assert "priority_level" in processed
        assert "client_tier" in processed


class TestSystemLoadMonitor:
    """Test cases for SystemLoadMonitor."""

    @pytest.mark.unit
    def test_monitor_initialization(self, test_config):
        """Test monitor initialization."""
        monitor = SystemLoadMonitor(test_config)

        assert monitor.config == test_config
        assert hasattr(monitor, 'load_history')
        assert hasattr(monitor, 'anomaly_detector')

    @pytest.mark.unit
    def test_get_current_load_mock(self, test_config):
        """Test getting current load (mock mode)."""
        # Disable monitoring to force mock mode
        config = test_config
        config.system.monitoring["log_system_metrics"] = False

        monitor = SystemLoadMonitor(config)
        load_metrics = monitor.get_current_load()

        assert isinstance(load_metrics, dict)
        assert "cpu_utilization" in load_metrics
        assert "memory_usage" in load_metrics
        assert "gpu_utilization" in load_metrics
        assert "timestamp" in load_metrics

        # Check value ranges
        for metric in ["cpu_utilization", "memory_usage", "gpu_utilization"]:
            assert 0 <= load_metrics[metric] <= 1

    @pytest.mark.unit
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.net_io_counters')
    @patch('psutil.disk_usage')
    @patch('psutil.pids')
    def test_get_current_load_real(
        self, mock_pids, mock_disk, mock_net, mock_memory, mock_cpu, test_config
    ):
        """Test getting current load (real mode)."""
        # Enable monitoring
        config = test_config
        config.system.monitoring["log_system_metrics"] = True

        # Mock psutil responses
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_net.return_value = MagicMock()
        mock_disk.return_value = MagicMock(used=50, total=100)
        mock_pids.return_value = list(range(100))

        monitor = SystemLoadMonitor(config)
        load_metrics = monitor.get_current_load()

        assert load_metrics["cpu_utilization"] == 0.5
        assert load_metrics["memory_usage"] == 0.6

    @pytest.mark.unit
    def test_load_history_update(self, test_config, setup_seeds):
        """Test load history updating."""
        monitor = SystemLoadMonitor(test_config)

        # Get some load metrics
        for _ in range(5):
            monitor.get_current_load()

        assert len(monitor.load_history) == 5

    @pytest.mark.unit
    def test_detect_load_anomaly(self, test_config, setup_seeds):
        """Test anomaly detection."""
        monitor = SystemLoadMonitor(test_config)

        # Generate normal load history
        for _ in range(100):
            monitor.get_current_load()

        # Test anomaly detection (should return False for unfitted detector)
        current_load = monitor.get_current_load()
        is_anomaly = monitor.detect_load_anomaly(current_load)
        assert isinstance(is_anomaly, bool)

    @pytest.mark.unit
    def test_get_load_trends(self, test_config, setup_seeds):
        """Test load trend analysis."""
        monitor = SystemLoadMonitor(test_config)

        # Generate load history
        for _ in range(50):
            monitor.get_current_load()

        trends = monitor.get_load_trends(window_size=30)

        # Check trend metrics
        assert isinstance(trends, dict)
        if trends:  # May be empty if insufficient data
            for metric in trends:
                assert isinstance(trends[metric], (int, float))

    @pytest.mark.unit
    def test_is_system_overloaded(self, test_config):
        """Test system overload detection."""
        monitor = SystemLoadMonitor(test_config)

        # Test with normal load
        normal_load = {
            "cpu_utilization": 0.5,
            "memory_usage": 0.6,
            "gpu_utilization": 0.4,
            "queue_length": 0.3,
        }
        assert not monitor.is_system_overloaded(normal_load)

        # Test with overload
        overload = {
            "cpu_utilization": 0.95,  # Above threshold
            "memory_usage": 0.6,
            "gpu_utilization": 0.4,
            "queue_length": 0.3,
        }
        assert monitor.is_system_overloaded(overload)