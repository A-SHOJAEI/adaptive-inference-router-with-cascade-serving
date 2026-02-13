"""
Adaptive Inference Router with Cascade Serving

A research-grade adaptive inference routing system that learns to dynamically
dispatch incoming requests across a cascade of model variants based on predicted
query difficulty, SLA constraints, and real-time cluster load.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .models.model import AdaptiveInferenceRouter, ModelCascade
from .training.trainer import MultiObjectiveTrainer
from .evaluation.metrics import RoutingMetrics, PerformanceTracker, StatisticalAnalyzer
from .utils.config import Config

__all__ = [
    "AdaptiveInferenceRouter",
    "ModelCascade",
    "MultiObjectiveTrainer",
    "RoutingMetrics",
    "PerformanceTracker",
    "StatisticalAnalyzer",
    "Config"
]