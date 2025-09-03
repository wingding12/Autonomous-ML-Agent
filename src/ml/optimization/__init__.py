"""
Optimization package for the Autonomous ML Agent.

This package contains hyperparameter optimization and
meta-learning components.
"""

from .hyperopt import HyperparameterOptimizer
from .meta_learning import MetaLearningOptimizer
from .search import SearchStrategy

__all__ = [
    "HyperparameterOptimizer",
    "MetaLearningOptimizer",
    "SearchStrategy"
]
