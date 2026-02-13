"""Data preprocessing utilities for adaptive inference routing."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.config import Config


logger = logging.getLogger(__name__)


class QueryDifficultyEstimator:
    """
    Estimates query difficulty based on input characteristics.

    Uses multiple heuristics and learned features to predict the computational
    complexity and resource requirements of incoming inference requests.
    """

    def __init__(self, config: Config) -> None:
        """Initialize query difficulty estimator.

        Args:
            config: Configuration containing model parameters.
        """
        self.config = config
        self.feature_weights: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info("Initialized QueryDifficultyEstimator")

    def fit(self, features: np.ndarray, latencies: np.ndarray) -> None:
        """Fit difficulty estimator on historical data.

        Args:
            features: Feature matrix [N, D] of query characteristics.
            latencies: Historical latency measurements [N] in milliseconds.
        """
        if features.shape[0] != latencies.shape[0]:
            raise ValueError("Features and latencies must have same number of samples")

        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        # Learn feature importance via correlation with latency
        correlations = np.corrcoef(features_normalized.T, latencies)[:-1, -1]
        self.feature_weights = np.abs(correlations)
        self.feature_weights /= np.sum(self.feature_weights)  # Normalize

        self.is_fitted = True
        logger.info(f"Fitted difficulty estimator on {features.shape[0]} samples")

    def estimate_difficulty(self, features: np.ndarray) -> np.ndarray:
        """Estimate query difficulty scores.

        Args:
            features: Query features [N, D].

        Returns:
            Difficulty scores [N] in range [0, 1].
        """
        if not self.is_fitted:
            # Use simple heuristics if not fitted
            return self._heuristic_difficulty(features)

        # Normalize features
        features_normalized = self.scaler.transform(features)

        # Compute weighted difficulty score
        difficulty_scores = np.dot(features_normalized, self.feature_weights)

        # Normalize to [0, 1] range
        difficulty_scores = (difficulty_scores - difficulty_scores.min()) / (
            difficulty_scores.max() - difficulty_scores.min() + 1e-8
        )

        return difficulty_scores

    def _heuristic_difficulty(self, features: np.ndarray) -> np.ndarray:
        """Compute heuristic difficulty when not fitted.

        Args:
            features: Query features [N, D].

        Returns:
            Heuristic difficulty scores [N].
        """
        # Simple heuristic based on feature magnitudes
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Assume first few features are complexity-related
        complexity_features = features[:, :min(3, features.shape[1])]
        difficulty_scores = np.mean(complexity_features, axis=1)

        # Normalize to [0, 1]
        difficulty_scores = np.clip(difficulty_scores / np.max(difficulty_scores), 0, 1)

        return difficulty_scores

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get learned feature importance weights.

        Returns:
            Feature importance weights or None if not fitted.
        """
        return self.feature_weights.copy() if self.is_fitted else None


class SLAConstraintProcessor:
    """
    Processes and validates SLA constraints for routing decisions.

    Handles different types of SLA constraints including latency bounds,
    accuracy requirements, and priority levels.
    """

    def __init__(self, config: Config) -> None:
        """Initialize SLA constraint processor.

        Args:
            config: Configuration containing SLA parameters.
        """
        self.config = config
        self.default_sla = self.config.environment.sla

        # SLA constraint types and their validation rules
        self.constraint_validators = {
            "target_latency_ms": self._validate_latency,
            "accuracy_threshold": self._validate_accuracy,
            "priority_level": self._validate_priority,
            "client_tier": self._validate_client_tier,
        }

        logger.info("Initialized SLAConstraintProcessor")

    def process_constraints(self, raw_constraints: Dict[str, Any]) -> Dict[str, float]:
        """Process and validate SLA constraints.

        Args:
            raw_constraints: Raw constraint dictionary from client.

        Returns:
            Processed and validated constraints.

        Raises:
            ValueError: If constraints are invalid.
        """
        processed_constraints = {}

        for constraint_name, value in raw_constraints.items():
            if constraint_name in self.constraint_validators:
                processed_value = self.constraint_validators[constraint_name](value)
                processed_constraints[constraint_name] = processed_value
            else:
                logger.warning(f"Unknown constraint type: {constraint_name}")

        # Add default constraints for missing values
        self._add_default_constraints(processed_constraints)

        return processed_constraints

    def _validate_latency(self, latency_ms: float) -> float:
        """Validate latency constraint.

        Args:
            latency_ms: Target latency in milliseconds.

        Returns:
            Validated latency constraint.

        Raises:
            ValueError: If latency is invalid.
        """
        if not isinstance(latency_ms, (int, float)):
            raise ValueError("Latency must be numeric")

        if latency_ms <= 0:
            raise ValueError("Latency must be positive")

        if latency_ms > 10000:  # 10 second max
            logger.warning(f"Very high latency constraint: {latency_ms}ms")

        return float(latency_ms)

    def _validate_accuracy(self, accuracy: float) -> float:
        """Validate accuracy constraint.

        Args:
            accuracy: Minimum accuracy requirement.

        Returns:
            Validated accuracy constraint.

        Raises:
            ValueError: If accuracy is invalid.
        """
        if not isinstance(accuracy, (int, float)):
            raise ValueError("Accuracy must be numeric")

        if not 0 <= accuracy <= 1:
            raise ValueError("Accuracy must be between 0 and 1")

        return float(accuracy)

    def _validate_priority(self, priority: int) -> float:
        """Validate priority level.

        Args:
            priority: Priority level (0=low, 1=normal, 2=high, 3=critical).

        Returns:
            Normalized priority score.

        Raises:
            ValueError: If priority is invalid.
        """
        if not isinstance(priority, int):
            raise ValueError("Priority must be integer")

        if not 0 <= priority <= 3:
            raise ValueError("Priority must be between 0 and 3")

        # Convert to normalized score
        return float(priority) / 3.0

    def _validate_client_tier(self, tier: int) -> float:
        """Validate client tier.

        Args:
            tier: Client tier (0=basic, 1=standard, 2=premium).

        Returns:
            Normalized tier score.

        Raises:
            ValueError: If tier is invalid.
        """
        if not isinstance(tier, int):
            raise ValueError("Client tier must be integer")

        if not 0 <= tier <= 2:
            raise ValueError("Client tier must be between 0 and 2")

        # Convert to normalized score
        return float(tier) / 2.0

    def _add_default_constraints(self, constraints: Dict[str, float]) -> None:
        """Add default values for missing constraints.

        Args:
            constraints: Constraint dictionary to update in-place.
        """
        defaults = {
            "target_latency_ms": self.default_sla.get("p95_latency_ms", 100),
            "accuracy_threshold": self.default_sla.get("min_accuracy", 0.95),
            "priority_level": 0.5,  # Normal priority
            "client_tier": 0.5,     # Standard tier
        }

        for key, default_value in defaults.items():
            if key not in constraints:
                constraints[key] = self._validate_by_type(key, default_value)

    def _validate_by_type(self, constraint_name: str, value: Any) -> float:
        """Validate constraint by type.

        Args:
            constraint_name: Name of the constraint.
            value: Value to validate.

        Returns:
            Validated constraint value.
        """
        if constraint_name in self.constraint_validators:
            return self.constraint_validators[constraint_name](value)
        else:
            return float(value)

    def check_sla_violation(
        self,
        predicted_performance: Dict[str, float],
        constraints: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if predicted performance violates SLA constraints.

        Args:
            predicted_performance: Predicted performance metrics.
            constraints: SLA constraints.

        Returns:
            Tuple of (is_violation, violation_details).
        """
        violations = {}
        is_violation = False

        # Check latency constraint
        if "latency_ms" in predicted_performance and "target_latency_ms" in constraints:
            predicted_latency = predicted_performance["latency_ms"]
            target_latency = constraints["target_latency_ms"]

            if predicted_latency > target_latency:
                violations["latency"] = predicted_latency - target_latency
                is_violation = True

        # Check accuracy constraint
        if "accuracy" in predicted_performance and "accuracy_threshold" in constraints:
            predicted_accuracy = predicted_performance["accuracy"]
            min_accuracy = constraints["accuracy_threshold"]

            if predicted_accuracy < min_accuracy:
                violations["accuracy"] = min_accuracy - predicted_accuracy
                is_violation = True

        return is_violation, violations


class SystemLoadMonitor:
    """
    Monitors real-time system load and resource utilization.

    Tracks CPU, GPU, memory, and network metrics to inform routing decisions
    based on current system capacity.
    """

    def __init__(self, config: Config) -> None:
        """Initialize system load monitor.

        Args:
            config: Configuration containing monitoring parameters.
        """
        self.config = config
        self.monitoring_enabled = config.system.monitoring.get("log_system_metrics", True)

        # Historical load tracking
        self.load_history: List[Dict[str, float]] = []
        self.max_history_size = 1000

        # Anomaly detection for load spikes
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_fitted = False

        logger.info("Initialized SystemLoadMonitor")

    def get_current_load(self) -> Dict[str, float]:
        """Get current system load metrics.

        Returns:
            Dictionary containing current system metrics.
        """
        if not self.monitoring_enabled:
            return self._get_mock_load()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU metrics (requires nvidia-ml-py)
            gpu_metrics = self._get_gpu_metrics()

            # Network metrics
            network = psutil.net_io_counters()
            network_sent_rate = getattr(self, '_last_network_sent', 0)
            network_recv_rate = getattr(self, '_last_network_recv', 0)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Queue length estimation (process count as proxy)
            queue_length = len(psutil.pids())

            load_metrics = {
                "cpu_utilization": cpu_percent / 100.0,
                "memory_usage": memory_percent / 100.0,
                "gpu_utilization": gpu_metrics.get("utilization", 0.0),
                "gpu_memory_usage": gpu_metrics.get("memory_usage", 0.0),
                "network_bandwidth": min(network_sent_rate + network_recv_rate, 1000.0) / 1000.0,
                "disk_usage": disk_percent / 100.0,
                "queue_length": min(queue_length, 1000) / 1000.0,
                "timestamp": time.time(),
            }

            # Store in history
            self._update_load_history(load_metrics)

            return load_metrics

        except Exception as e:
            logger.error(f"Error getting system load: {e}")
            return self._get_mock_load()

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics.

        Returns:
            GPU metrics dictionary.
        """
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return {"utilization": 0.0, "memory_usage": 0.0}

            # Get metrics for first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return {
                "utilization": gpu_util.gpu / 100.0,
                "memory_usage": memory_info.used / memory_info.total,
            }

        except ImportError:
            # pynvml not available
            return {"utilization": 0.0, "memory_usage": 0.0}
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            return {"utilization": 0.0, "memory_usage": 0.0}

    def _get_mock_load(self) -> Dict[str, float]:
        """Get mock load metrics for testing.

        Returns:
            Mock system metrics.
        """
        # Generate realistic but synthetic load patterns
        base_load = 0.3 + 0.4 * np.sin(time.time() / 3600)  # Hourly pattern
        noise = np.random.normal(0, 0.1)

        mock_load = base_load + noise
        mock_load = np.clip(mock_load, 0.1, 0.9)

        return {
            "cpu_utilization": mock_load,
            "memory_usage": mock_load * 0.8,
            "gpu_utilization": mock_load * 1.2,
            "gpu_memory_usage": mock_load * 0.9,
            "network_bandwidth": mock_load * 0.6,
            "disk_usage": 0.5,
            "queue_length": mock_load * 0.3,
            "timestamp": time.time(),
        }

    def _update_load_history(self, load_metrics: Dict[str, float]) -> None:
        """Update load history for trend analysis.

        Args:
            load_metrics: Current load metrics to add to history.
        """
        # Add to history
        self.load_history.append(load_metrics.copy())

        # Trim history if too large
        if len(self.load_history) > self.max_history_size:
            self.load_history = self.load_history[-self.max_history_size:]

        # Update anomaly detector periodically
        if len(self.load_history) % 100 == 0 and len(self.load_history) >= 100:
            self._fit_anomaly_detector()

    def _fit_anomaly_detector(self) -> None:
        """Fit anomaly detector on historical load data."""
        if len(self.load_history) < 50:
            return

        # Extract numeric features for anomaly detection
        features = []
        for load in self.load_history:
            feature_vector = [
                load["cpu_utilization"],
                load["memory_usage"],
                load["gpu_utilization"],
                load["queue_length"],
            ]
            features.append(feature_vector)

        features_array = np.array(features)

        try:
            self.anomaly_detector.fit(features_array)
            self.anomaly_fitted = True
        except Exception as e:
            logger.warning(f"Error fitting anomaly detector: {e}")

    def detect_load_anomaly(self, load_metrics: Dict[str, float]) -> bool:
        """Detect if current load is anomalous.

        Args:
            load_metrics: Current load metrics.

        Returns:
            True if load is anomalous, False otherwise.
        """
        if not self.anomaly_fitted:
            return False

        try:
            feature_vector = np.array([[
                load_metrics["cpu_utilization"],
                load_metrics["memory_usage"],
                load_metrics["gpu_utilization"],
                load_metrics["queue_length"],
            ]])

            prediction = self.anomaly_detector.predict(feature_vector)
            return prediction[0] == -1  # -1 indicates anomaly

        except Exception as e:
            logger.warning(f"Error detecting anomaly: {e}")
            return False

    def get_load_trends(self, window_size: int = 100) -> Dict[str, float]:
        """Get load trend analysis over recent history.

        Args:
            window_size: Number of recent samples to analyze.

        Returns:
            Dictionary containing trend metrics.
        """
        if len(self.load_history) < window_size:
            window_size = len(self.load_history)

        if window_size < 2:
            return {}

        recent_history = self.load_history[-window_size:]

        trends = {}
        for metric in ["cpu_utilization", "memory_usage", "gpu_utilization", "queue_length"]:
            values = [load[metric] for load in recent_history]

            # Calculate trend slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            trends[f"{metric}_trend"] = slope
            trends[f"{metric}_mean"] = np.mean(values)
            trends[f"{metric}_std"] = np.std(values)

        return trends

    def is_system_overloaded(self, load_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Check if system is currently overloaded.

        Args:
            load_metrics: Current load metrics (gets current if None).

        Returns:
            True if system is overloaded, False otherwise.
        """
        if load_metrics is None:
            load_metrics = self.get_current_load()

        # Define overload thresholds
        overload_thresholds = {
            "cpu_utilization": 0.9,
            "memory_usage": 0.9,
            "gpu_utilization": 0.95,
            "queue_length": 0.8,
        }

        # Check if any metric exceeds threshold
        for metric, threshold in overload_thresholds.items():
            if load_metrics.get(metric, 0) > threshold:
                logger.warning(f"System overload detected: {metric}={load_metrics[metric]:.3f}")
                return True

        return False