"""Data loading utilities for adaptive inference routing."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..utils.config import Config


logger = logging.getLogger(__name__)


class MLPerfInferenceDataset(Dataset):
    """
    MLPerf Inference benchmark dataset for training adaptive routing policies.

    This dataset simulates realistic inference requests with varying difficulty levels,
    SLA constraints, and system load conditions based on MLPerf Inference patterns.
    """

    def __init__(
        self,
        size: int = 10000,
        difficulty_features: Optional[List[str]] = None,
        sla_features: Optional[List[str]] = None,
        system_features: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize MLPerf inference dataset.

        Args:
            size: Number of synthetic inference requests to generate.
            difficulty_features: List of query difficulty feature names.
            sla_features: List of SLA constraint feature names.
            system_features: List of system load feature names.
            seed: Random seed for reproducible generation.
        """
        self.size = size
        self.seed = seed

        # Default feature sets based on MLPerf characteristics
        self.difficulty_features = difficulty_features or [
            "input_complexity",
            "computational_graph_depth",
            "memory_footprint",
            "numerical_stability",
            "inference_uncertainty",
        ]

        self.sla_features = sla_features or [
            "target_latency_ms",
            "accuracy_threshold",
            "priority_level",
            "client_tier",
        ]

        self.system_features = system_features or [
            "cpu_utilization",
            "gpu_utilization",
            "memory_usage",
            "network_bandwidth",
            "queue_length",
        ]

        # Generate synthetic dataset
        self._generate_synthetic_data()

        logger.info(
            f"Generated MLPerf dataset with {self.size} samples, "
            f"{len(self.difficulty_features)} difficulty features, "
            f"{len(self.sla_features)} SLA features, "
            f"{len(self.system_features)} system features"
        )

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic inference request data."""
        np.random.seed(self.seed)

        # Generate difficulty features (normalized)
        difficulty_data = {}
        for feature in self.difficulty_features:
            if feature == "input_complexity":
                # Log-normal distribution for input complexity
                difficulty_data[feature] = np.random.lognormal(0, 1, self.size)
            elif feature == "computational_graph_depth":
                # Exponential distribution for graph depth
                difficulty_data[feature] = np.random.exponential(2, self.size)
            elif feature == "memory_footprint":
                # Gamma distribution for memory usage
                difficulty_data[feature] = np.random.gamma(2, 2, self.size)
            elif feature == "numerical_stability":
                # Beta distribution for stability score
                difficulty_data[feature] = np.random.beta(2, 2, self.size)
            else:  # inference_uncertainty
                # Uniform distribution for uncertainty
                difficulty_data[feature] = np.random.uniform(0, 1, self.size)

        # Generate SLA constraint features
        sla_data = {}
        for feature in self.sla_features:
            if feature == "target_latency_ms":
                # Multi-modal distribution for latency SLAs
                mode = np.random.choice([50, 100, 200, 500], self.size, p=[0.2, 0.4, 0.3, 0.1])
                sla_data[feature] = mode + np.random.normal(0, 10, self.size)
            elif feature == "accuracy_threshold":
                # High accuracy requirements
                sla_data[feature] = np.random.uniform(0.85, 0.99, self.size)
            elif feature == "priority_level":
                # Categorical priority levels
                sla_data[feature] = np.random.choice([0, 1, 2, 3], self.size, p=[0.1, 0.3, 0.4, 0.2])
            else:  # client_tier
                # Client tier (premium, standard, basic)
                sla_data[feature] = np.random.choice([0, 1, 2], self.size, p=[0.2, 0.5, 0.3])

        # Generate system load features
        system_data = {}
        for feature in self.system_features:
            if feature in ["cpu_utilization", "gpu_utilization", "memory_usage"]:
                # Correlated utilization metrics with temporal patterns
                base_load = np.random.uniform(0.3, 0.9, self.size)
                noise = np.random.normal(0, 0.1, self.size)
                system_data[feature] = np.clip(base_load + noise, 0, 1)
            elif feature == "network_bandwidth":
                # Exponential distribution for bandwidth
                system_data[feature] = np.random.exponential(100, self.size)
            else:  # queue_length
                # Poisson distribution for queue length
                system_data[feature] = np.random.poisson(5, self.size)

        # Combine all features
        self.features = {**difficulty_data, **sla_data, **system_data}
        self.feature_names = list(self.features.keys())

        # Generate ground truth optimal routing decisions
        self._generate_optimal_routes()

    def _generate_optimal_routes(self) -> None:
        """Generate ground truth optimal routing decisions."""
        # Simulate optimal routing policy based on feature combinations
        optimal_routes = []

        for i in range(self.size):
            # Extract features for this sample
            complexity = self.features["input_complexity"][i]
            latency_sla = self.features["target_latency_ms"][i]
            accuracy_sla = self.features["accuracy_threshold"][i]
            system_load = self.features["cpu_utilization"][i]

            # Simple heuristic for optimal routing (would be learned in practice)
            if latency_sla < 75 and accuracy_sla < 0.92:
                route = 0  # quantized_int8
            elif latency_sla < 150 and complexity < 2.0:
                route = 1  # pruned_50
            elif accuracy_sla < 0.96 and system_load > 0.7:
                route = 2  # distilled
            else:
                route = 3  # full_precision

            optimal_routes.append(route)

        self.optimal_routes = np.array(optimal_routes)

        # Generate performance metrics for each route choice
        self._generate_performance_metrics()

    def _generate_performance_metrics(self) -> None:
        """Generate realistic performance metrics for routing decisions."""
        # Model cascade characteristics (from config defaults)
        cascade_configs = [
            {"latency_mult": 0.3, "accuracy_mult": 0.95, "memory_mult": 0.25},  # quantized
            {"latency_mult": 0.5, "accuracy_mult": 0.97, "memory_mult": 0.5},   # pruned
            {"latency_mult": 0.4, "accuracy_mult": 0.94, "memory_mult": 0.3},   # distilled
            {"latency_mult": 1.0, "accuracy_mult": 1.0, "memory_mult": 1.0},    # full
        ]

        # Generate baseline metrics
        base_latency = 100 + self.features["input_complexity"] * 50
        base_accuracy = 0.95 - self.features["inference_uncertainty"] * 0.1
        base_memory = 1000 + self.features["memory_footprint"] * 200

        self.performance_metrics = {}
        for route_idx, config in enumerate(cascade_configs):
            route_name = f"route_{route_idx}"

            # Apply model-specific multipliers with noise
            latency_noise = np.random.normal(1.0, 0.1, self.size)
            accuracy_noise = np.random.normal(1.0, 0.02, self.size)
            memory_noise = np.random.normal(1.0, 0.05, self.size)

            self.performance_metrics[f"{route_name}_latency"] = (
                base_latency * config["latency_mult"] * latency_noise
            )
            self.performance_metrics[f"{route_name}_accuracy"] = np.clip(
                base_accuracy * config["accuracy_mult"] * accuracy_noise, 0, 1
            )
            self.performance_metrics[f"{route_name}_memory"] = (
                base_memory * config["memory_mult"] * memory_noise
            )

    def __len__(self) -> int:
        """Get dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing features, optimal route, and performance metrics.
        """
        # Extract features
        feature_vector = np.array([self.features[name][idx] for name in self.feature_names])

        # Extract performance metrics for all routes
        performance_vector = np.array([
            self.performance_metrics[f"route_{i}_latency"][idx]
            for i in range(4)
        ])

        return {
            "features": torch.tensor(feature_vector, dtype=torch.float32),
            "optimal_route": torch.tensor(self.optimal_routes[idx], dtype=torch.long),
            "performance": torch.tensor(performance_vector, dtype=torch.float32),
            "sla_constraints": torch.tensor([
                self.features["target_latency_ms"][idx],
                self.features["accuracy_threshold"][idx],
            ], dtype=torch.float32),
        }

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "size": self.size,
            "num_features": len(self.feature_names),
            "route_distribution": np.bincount(self.optimal_routes),
        }

        # Feature statistics
        for name in self.feature_names:
            data = self.features[name]
            stats[f"{name}_mean"] = np.mean(data)
            stats[f"{name}_std"] = np.std(data)
            stats[f"{name}_min"] = np.min(data)
            stats[f"{name}_max"] = np.max(data)

        return stats


class AdaptiveDataLoader:
    """
    Adaptive data loader for training routing policies.

    Manages data loading, preprocessing, and splits for training the
    multi-objective reinforcement learning routing system.
    """

    def __init__(self, config: Config) -> None:
        """Initialize adaptive data loader.

        Args:
            config: Configuration containing data parameters.
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

        logger.info("Initialized AdaptiveDataLoader")

    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        # Create full dataset
        dataset_size = 50000  # Large enough for research evaluation
        full_dataset = MLPerfInferenceDataset(
            size=dataset_size,
            difficulty_features=self.config.data.difficulty_features,
            sla_features=self.config.data.sla_features,
            system_features=self.config.data.system_features,
            seed=self.config.system.seed,
        )

        # Calculate split sizes
        train_size = int(self.config.data.train_split * dataset_size)
        val_size = int(self.config.data.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Create splits
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.system.seed)
        )

        logger.info(
            f"Created datasets: train={train_size}, val={val_size}, test={test_size}"
        )

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        # Training loader with shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True,
        )

        # Validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

        # Test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

        logger.info(
            f"Created data loaders: "
            f"train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}, "
            f"test_batches={len(test_loader)}"
        )

        return train_loader, val_loader, test_loader

    def setup_preprocessing(self, train_dataset: Dataset) -> None:
        """Setup preprocessing based on training data statistics.

        Args:
            train_dataset: Training dataset for computing statistics.
        """
        # Extract features from training data for normalization
        train_features = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            train_features.append(sample["features"].numpy())

        train_features = np.array(train_features)

        # Fit scalers on training data
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_features)
        self.scalers["features"] = feature_scaler

        logger.info(f"Setup preprocessing with feature scaler")

    def preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply preprocessing to a batch of data.

        Args:
            batch: Batch of data to preprocess.

        Returns:
            Preprocessed batch.
        """
        processed_batch = batch.copy()

        # Normalize features if scaler is available
        if "features" in self.scalers:
            features = batch["features"].numpy()
            features_normalized = self.scalers["features"].transform(features)
            processed_batch["features"] = torch.tensor(features_normalized, dtype=torch.float32)

        return processed_batch

    def get_data_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """Get comprehensive data statistics.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Dictionary containing data statistics.
        """
        if isinstance(dataset.dataset, MLPerfInferenceDataset):
            base_stats = dataset.dataset.get_statistics()
        else:
            # For subset datasets, compute basic statistics
            base_stats = {"size": len(dataset)}

        return base_stats

    def save_preprocessing_state(self, save_path: Union[str, Path]) -> None:
        """Save preprocessing state for later use.

        Args:
            save_path: Path to save preprocessing state.
        """
        import pickle

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "scalers": self.scalers,
            "encoders": self.encoders,
        }

        with save_path.open("wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved preprocessing state to {save_path}")

    def load_preprocessing_state(self, load_path: Union[str, Path]) -> None:
        """Load preprocessing state.

        Args:
            load_path: Path to load preprocessing state from.
        """
        import pickle

        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessing state not found: {load_path}")

        with load_path.open("rb") as f:
            state = pickle.load(f)

        self.scalers = state["scalers"]
        self.encoders = state["encoders"]

        logger.info(f"Loaded preprocessing state from {load_path}")