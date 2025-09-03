"""
Evaluation package for the Autonomous ML Agent.

This package contains model evaluation, metrics calculation,
and interpretability components.
"""

from .evaluator import ModelEvaluator
from .metrics import MetricsCalculator
from .interpretability import InterpretabilityAnalyzer

__all__ = [
    "ModelEvaluator",
    "MetricsCalculator",
    "InterpretabilityAnalyzer"
]
