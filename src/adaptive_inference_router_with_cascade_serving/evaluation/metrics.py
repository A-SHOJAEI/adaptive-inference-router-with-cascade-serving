"""
Evaluation metrics and performance tracking for adaptive inference routing.

This module provides comprehensive evaluation metrics for the multi-objective
routing system including statistical significance testing and ablation studies.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.config import Config


logger = logging.getLogger(__name__)


class RoutingMetrics:
    """
    Comprehensive metrics for evaluating routing performance.

    Tracks the key performance indicators for the adaptive routing system
    including latency, accuracy, throughput, and SLA compliance.
    """

    def __init__(self, config: Config) -> None:
        """Initialize routing metrics.

        Args:
            config: Configuration containing evaluation parameters.
        """
        self.config = config
        self.target_metrics = config.evaluation.target_metrics

        # Metric storage
        self.reset()

        logger.info("Initialized RoutingMetrics")

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.routing_decisions: List[int] = []
        self.optimal_decisions: List[int] = []
        self.latencies: List[float] = []
        self.accuracies: List[float] = []
        self.throughputs: List[float] = []
        self.sla_violations: List[bool] = []
        self.prediction_errors: List[float] = []
        self.timestamps: List[float] = []

        # Performance tracking
        self.route_performance: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def update(
        self,
        routing_decision: int,
        optimal_decision: int,
        latency: float,
        accuracy: float,
        throughput: float,
        sla_violated: bool,
        prediction_error: float
    ) -> None:
        """Update metrics with new measurements.

        Args:
            routing_decision: Predicted routing decision.
            optimal_decision: Optimal routing decision.
            latency: Measured latency in milliseconds.
            accuracy: Measured accuracy.
            throughput: Measured throughput in RPS.
            sla_violated: Whether SLA was violated.
            prediction_error: Performance prediction error.
        """
        current_time = time.time()

        self.routing_decisions.append(routing_decision)
        self.optimal_decisions.append(optimal_decision)
        self.latencies.append(latency)
        self.accuracies.append(accuracy)
        self.throughputs.append(throughput)
        self.sla_violations.append(sla_violated)
        self.prediction_errors.append(prediction_error)
        self.timestamps.append(current_time)

        # Per-route performance tracking
        self.route_performance[routing_decision]["latency"].append(latency)
        self.route_performance[routing_decision]["accuracy"].append(accuracy)
        self.route_performance[routing_decision]["throughput"].append(throughput)

    def compute_routing_accuracy(self) -> float:
        """Compute routing decision accuracy.

        Returns:
            Routing accuracy as percentage.
        """
        if not self.routing_decisions:
            return 0.0

        correct = sum(
            r == o for r, o in zip(self.routing_decisions, self.optimal_decisions)
        )
        return correct / len(self.routing_decisions)

    def compute_latency_metrics(self) -> Dict[str, float]:
        """Compute latency performance metrics.

        Returns:
            Dictionary containing latency statistics.
        """
        if not self.latencies:
            return {}

        latencies_array = np.array(self.latencies)

        return {
            "mean_latency": np.mean(latencies_array),
            "p50_latency": np.percentile(latencies_array, 50),
            "p95_latency": np.percentile(latencies_array, 95),
            "p99_latency": np.percentile(latencies_array, 99),
            "max_latency": np.max(latencies_array),
            "std_latency": np.std(latencies_array),
        }

    def compute_accuracy_metrics(self) -> Dict[str, float]:
        """Compute accuracy performance metrics.

        Returns:
            Dictionary containing accuracy statistics.
        """
        if not self.accuracies:
            return {}

        accuracies_array = np.array(self.accuracies)

        return {
            "mean_accuracy": np.mean(accuracies_array),
            "min_accuracy": np.min(accuracies_array),
            "std_accuracy": np.std(accuracies_array),
            "accuracy_degradation": max(0, 1.0 - np.mean(accuracies_array)),
        }

    def compute_throughput_metrics(self) -> Dict[str, float]:
        """Compute throughput performance metrics.

        Returns:
            Dictionary containing throughput statistics.
        """
        if not self.throughputs:
            return {}

        throughputs_array = np.array(self.throughputs)

        return {
            "mean_throughput": np.mean(throughputs_array),
            "max_throughput": np.max(throughputs_array),
            "std_throughput": np.std(throughputs_array),
            "throughput_improvement": max(0, np.mean(throughputs_array) - 50.0),  # vs baseline
        }

    def compute_sla_metrics(self) -> Dict[str, float]:
        """Compute SLA compliance metrics.

        Returns:
            Dictionary containing SLA statistics.
        """
        if not self.sla_violations:
            return {}

        violation_rate = np.mean(self.sla_violations)

        return {
            "sla_violation_rate": violation_rate,
            "sla_compliance_rate": 1.0 - violation_rate,
            "total_violations": sum(self.sla_violations),
        }

    def compute_prediction_accuracy(self) -> Dict[str, float]:
        """Compute performance prediction accuracy.

        Returns:
            Dictionary containing prediction accuracy metrics.
        """
        if not self.prediction_errors:
            return {}

        errors_array = np.array(self.prediction_errors)

        return {
            "mean_prediction_error": np.mean(errors_array),
            "rmse_prediction_error": np.sqrt(np.mean(errors_array ** 2)),
            "max_prediction_error": np.max(errors_array),
        }

    def compute_route_distribution(self) -> Dict[str, float]:
        """Compute routing decision distribution.

        Returns:
            Dictionary containing route usage statistics.
        """
        if not self.routing_decisions:
            return {}

        route_counts = {}
        total_decisions = len(self.routing_decisions)

        for route_id in set(self.routing_decisions):
            count = self.routing_decisions.count(route_id)
            route_counts[f"route_{route_id}_usage"] = count / total_decisions

        return route_counts

    def compute_target_metrics(self) -> Dict[str, float]:
        """Compute target research metrics.

        Returns:
            Dictionary containing target metrics for research evaluation.
        """
        latency_metrics = self.compute_latency_metrics()
        accuracy_metrics = self.compute_accuracy_metrics()
        throughput_metrics = self.compute_throughput_metrics()
        sla_metrics = self.compute_sla_metrics()

        # Compute metrics relative to targets
        target_metrics = {}

        # P99 latency reduction vs static routing (simulated baseline)
        if "p99_latency" in latency_metrics:
            baseline_p99 = 150.0  # Simulated static routing baseline
            current_p99 = latency_metrics["p99_latency"]
            reduction = max(0, (baseline_p99 - current_p99) / baseline_p99)
            target_metrics["p99_latency_reduction_vs_static"] = reduction

        # Accuracy degradation vs full model
        if "accuracy_degradation" in accuracy_metrics:
            target_metrics["accuracy_degradation_vs_full_model"] = accuracy_metrics["accuracy_degradation"]

        # Throughput improvement in RPS
        if "throughput_improvement" in throughput_metrics:
            target_metrics["throughput_improvement_rps"] = throughput_metrics["throughput_improvement"] / 50.0

        # SLA violation rate
        if "sla_violation_rate" in sla_metrics:
            target_metrics["sla_violation_rate"] = sla_metrics["sla_violation_rate"]

        return target_metrics

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Dictionary containing all computed metrics.
        """
        return {
            "routing_accuracy": self.compute_routing_accuracy(),
            "latency_metrics": self.compute_latency_metrics(),
            "accuracy_metrics": self.compute_accuracy_metrics(),
            "throughput_metrics": self.compute_throughput_metrics(),
            "sla_metrics": self.compute_sla_metrics(),
            "prediction_metrics": self.compute_prediction_accuracy(),
            "route_distribution": self.compute_route_distribution(),
            "target_metrics": self.compute_target_metrics(),
            "summary_stats": {
                "total_requests": len(self.routing_decisions),
                "evaluation_duration": self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0,
            }
        }

    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix for routing decisions.

        Returns:
            Confusion matrix as numpy array.
        """
        if not self.routing_decisions or not self.optimal_decisions:
            return np.array([])

        num_routes = max(max(self.routing_decisions), max(self.optimal_decisions)) + 1
        return confusion_matrix(
            self.optimal_decisions,
            self.routing_decisions,
            labels=list(range(num_routes))
        )

    def generate_classification_report(self) -> str:
        """Generate detailed classification report.

        Returns:
            Classification report as string.
        """
        if not self.routing_decisions or not self.optimal_decisions:
            return "No data available for classification report"

        return classification_report(
            self.optimal_decisions,
            self.routing_decisions,
            target_names=[f"Route_{i}" for i in range(4)],
            zero_division=0
        )


class PerformanceTracker:
    """
    Real-time performance tracking for the routing system.

    Tracks performance metrics over time with sliding window analysis
    and trend detection for online monitoring.
    """

    def __init__(self, window_size: int = 1000, alert_thresholds: Optional[Dict[str, float]] = None) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Size of sliding window for metrics.
            alert_thresholds: Thresholds for performance alerts.
        """
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            "latency_p99": 200.0,
            "sla_violation_rate": 0.05,
            "accuracy_degradation": 0.1,
        }

        # Sliding windows
        self.latency_window = deque(maxlen=window_size)
        self.accuracy_window = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=window_size)
        self.sla_violation_window = deque(maxlen=window_size)

        # Trend tracking
        self.trend_history: Dict[str, deque] = {
            "latency": deque(maxlen=50),
            "accuracy": deque(maxlen=50),
            "throughput": deque(maxlen=50),
        }

        logger.info(f"Initialized PerformanceTracker with window_size={window_size}")

    def update(
        self,
        latency: float,
        accuracy: float,
        throughput: float,
        sla_violated: bool
    ) -> None:
        """Update performance metrics.

        Args:
            latency: Request latency in milliseconds.
            accuracy: Inference accuracy.
            throughput: Current throughput in RPS.
            sla_violated: Whether SLA was violated.
        """
        # Update sliding windows
        self.latency_window.append(latency)
        self.accuracy_window.append(accuracy)
        self.throughput_window.append(throughput)
        self.sla_violation_window.append(sla_violated)

        # Update trends (compute means periodically)
        if len(self.latency_window) % 20 == 0:
            self.trend_history["latency"].append(np.mean(list(self.latency_window)[-20:]))
            self.trend_history["accuracy"].append(np.mean(list(self.accuracy_window)[-20:]))
            self.trend_history["throughput"].append(np.mean(list(self.throughput_window)[-20:]))

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics.

        Returns:
            Dictionary containing current performance metrics.
        """
        metrics = {}

        if self.latency_window:
            latencies = np.array(self.latency_window)
            metrics.update({
                "current_mean_latency": np.mean(latencies),
                "current_p95_latency": np.percentile(latencies, 95),
                "current_p99_latency": np.percentile(latencies, 99),
            })

        if self.accuracy_window:
            accuracies = np.array(self.accuracy_window)
            metrics.update({
                "current_mean_accuracy": np.mean(accuracies),
                "current_min_accuracy": np.min(accuracies),
            })

        if self.throughput_window:
            throughputs = np.array(self.throughput_window)
            metrics.update({
                "current_mean_throughput": np.mean(throughputs),
                "current_max_throughput": np.max(throughputs),
            })

        if self.sla_violation_window:
            violations = np.array(self.sla_violation_window)
            metrics.update({
                "current_sla_violation_rate": np.mean(violations),
            })

        return metrics

    def detect_performance_degradation(self) -> Dict[str, bool]:
        """Detect performance degradation based on trends.

        Returns:
            Dictionary indicating which metrics are degrading.
        """
        degradation_alerts = {}

        # Analyze trends
        for metric_name, history in self.trend_history.items():
            if len(history) < 10:
                degradation_alerts[f"{metric_name}_degrading"] = False
                continue

            # Compute trend slope
            x = np.arange(len(history))
            slope, _, _, p_value, _ = stats.linregress(x, list(history))

            # Check for significant degradation
            if metric_name == "latency":
                degradation_alerts[f"{metric_name}_degrading"] = slope > 0 and p_value < 0.05
            elif metric_name in ["accuracy", "throughput"]:
                degradation_alerts[f"{metric_name}_degrading"] = slope < 0 and p_value < 0.05

        return degradation_alerts

    def check_alerts(self) -> List[str]:
        """Check for performance alerts.

        Returns:
            List of alert messages.
        """
        alerts = []
        current_metrics = self.get_current_metrics()

        # Check latency alerts
        if "current_p99_latency" in current_metrics:
            if current_metrics["current_p99_latency"] > self.alert_thresholds["latency_p99"]:
                alerts.append(
                    f"High P99 latency: {current_metrics['current_p99_latency']:.1f}ms "
                    f"(threshold: {self.alert_thresholds['latency_p99']:.1f}ms)"
                )

        # Check SLA violation rate
        if "current_sla_violation_rate" in current_metrics:
            if current_metrics["current_sla_violation_rate"] > self.alert_thresholds["sla_violation_rate"]:
                alerts.append(
                    f"High SLA violation rate: {current_metrics['current_sla_violation_rate']:.3f} "
                    f"(threshold: {self.alert_thresholds['sla_violation_rate']:.3f})"
                )

        # Check accuracy degradation
        if "current_mean_accuracy" in current_metrics:
            baseline_accuracy = 0.95  # Assumed baseline
            degradation = max(0, baseline_accuracy - current_metrics["current_mean_accuracy"])
            if degradation > self.alert_thresholds["accuracy_degradation"]:
                alerts.append(
                    f"Accuracy degradation: {degradation:.3f} "
                    f"(threshold: {self.alert_thresholds['accuracy_degradation']:.3f})"
                )

        return alerts

    def get_trend_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get trend analysis for all metrics.

        Returns:
            Dictionary containing trend statistics.
        """
        trend_analysis = {}

        for metric_name, history in self.trend_history.items():
            if len(history) < 5:
                trend_analysis[metric_name] = {"slope": 0.0, "r_value": 0.0, "p_value": 1.0}
                continue

            x = np.arange(len(history))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, list(history))

            trend_analysis[metric_name] = {
                "slope": slope,
                "r_value": r_value,
                "p_value": p_value,
                "std_err": std_err,
                "is_significant": p_value < 0.05,
            }

        return trend_analysis


class StatisticalAnalyzer:
    """
    Statistical analysis tools for routing evaluation.

    Provides statistical significance testing, confidence intervals,
    and comparative analysis for ablation studies.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize statistical analyzer.

        Args:
            alpha: Significance level for statistical tests.
        """
        self.alpha = alpha

        logger.info(f"Initialized StatisticalAnalyzer with alpha={alpha}")

    def compare_routing_strategies(
        self,
        strategy_a_metrics: List[float],
        strategy_b_metrics: List[float],
        metric_name: str = "latency"
    ) -> Dict[str, Any]:
        """Compare two routing strategies statistically.

        Args:
            strategy_a_metrics: Metrics for strategy A.
            strategy_b_metrics: Metrics for strategy B.
            metric_name: Name of the metric being compared.

        Returns:
            Dictionary containing statistical comparison results.
        """
        if not strategy_a_metrics or not strategy_b_metrics:
            return {"error": "Insufficient data for comparison"}

        # Basic statistics
        a_mean, a_std = np.mean(strategy_a_metrics), np.std(strategy_a_metrics)
        b_mean, b_std = np.mean(strategy_b_metrics), np.std(strategy_b_metrics)

        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(
            strategy_a_metrics, strategy_b_metrics, equal_var=False
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(strategy_a_metrics) - 1) * a_std**2 +
                             (len(strategy_b_metrics) - 1) * b_std**2) /
                            (len(strategy_a_metrics) + len(strategy_b_metrics) - 2))
        cohens_d = (a_mean - b_mean) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for difference
        diff_mean = a_mean - b_mean
        diff_se = np.sqrt(a_std**2/len(strategy_a_metrics) + b_std**2/len(strategy_b_metrics))
        t_critical = stats.t.ppf(1 - self.alpha/2,
                               len(strategy_a_metrics) + len(strategy_b_metrics) - 2)
        ci_lower = diff_mean - t_critical * diff_se
        ci_upper = diff_mean + t_critical * diff_se

        return {
            "metric_name": metric_name,
            "strategy_a_mean": a_mean,
            "strategy_a_std": a_std,
            "strategy_b_mean": b_mean,
            "strategy_b_std": b_std,
            "difference": diff_mean,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "effect_size_cohens_d": cohens_d,
            "confidence_interval": (ci_lower, ci_upper),
            "sample_sizes": (len(strategy_a_metrics), len(strategy_b_metrics)),
        }

    def compute_confidence_interval(
        self,
        data: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute confidence interval for data.

        Args:
            data: Data points.
            confidence_level: Confidence level (default 95%).

        Returns:
            Tuple of (mean, lower_bound, upper_bound).
        """
        if not data:
            return 0.0, 0.0, 0.0

        data_array = np.array(data)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, len(data) - 1)

        margin_error = t_critical * std_err
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error

        return mean, lower_bound, upper_bound

    def perform_anova(
        self,
        groups: Dict[str, List[float]],
        metric_name: str = "performance"
    ) -> Dict[str, Any]:
        """Perform one-way ANOVA across multiple groups.

        Args:
            groups: Dictionary mapping group names to metric lists.
            metric_name: Name of the metric being analyzed.

        Returns:
            Dictionary containing ANOVA results.
        """
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}

        group_data = [np.array(data) for data in groups.values() if data]
        if len(group_data) < 2:
            return {"error": "Need at least 2 non-empty groups for ANOVA"}

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)

        # Compute group statistics
        group_stats = {}
        for name, data in groups.items():
            if data:
                group_stats[name] = {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "count": len(data)
                }

        # Post-hoc analysis (Tukey HSD would require additional library)
        pairwise_comparisons = {}
        group_names = list(groups.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name_a, name_b = group_names[i], group_names[j]
                if groups[name_a] and groups[name_b]:
                    comparison = self.compare_routing_strategies(
                        groups[name_a], groups[name_b], f"{name_a}_vs_{name_b}"
                    )
                    pairwise_comparisons[f"{name_a}_vs_{name_b}"] = comparison

        return {
            "metric_name": metric_name,
            "f_statistic": f_stat,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "group_statistics": group_stats,
            "pairwise_comparisons": pairwise_comparisons,
            "total_groups": len(group_data),
        }

    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_func: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval.

        Args:
            data: Original data.
            statistic_func: Function to compute statistic.
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level.

        Returns:
            Tuple of (original_statistic, lower_bound, upper_bound).
        """
        if not data:
            return 0.0, 0.0, 0.0

        data_array = np.array(data)
        original_stat = statistic_func(data_array)

        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))

        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)

        return original_stat, lower_bound, upper_bound

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute statistical power for given parameters.

        Args:
            effect_size: Expected effect size (Cohen's d).
            sample_size: Sample size per group.
            alpha: Significance level (uses instance alpha if None).

        Returns:
            Dictionary containing power analysis results.
        """
        if alpha is None:
            alpha = self.alpha

        # This is a simplified power calculation
        # In practice, you might use statsmodels.stats.power for more precise calculations
        from math import sqrt
        from scipy.stats import norm

        # Critical value
        z_alpha = norm.ppf(1 - alpha/2)

        # Non-centrality parameter
        ncp = effect_size * sqrt(sample_size / 2)

        # Power calculation
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

        return {
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "power": power,
            "beta": 1 - power,
        }