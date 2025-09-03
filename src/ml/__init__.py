"""
Machine Learning package for the Autonomous ML Agent.

This package contains all ML-related functionality including models,
preprocessing, optimization, and evaluation components.
"""

from .models import *
from .preprocessing import *
from .optimization import *
from .evaluation import *

__all__ = [
    # Models
    "BaseModel",
    "ClassifierModel",
    "RegressorModel",
    "EnsembleModel",
    
    # Preprocessing
    "DataCleaner",
    "FeatureEncoder",
    "FeatureScaler",
    
    # Optimization
    "HyperparameterOptimizer",
    "MetaLearningOptimizer",
    
    # Evaluation
    "ModelEvaluator",
    "MetricsCalculator",
    "InterpretabilityAnalyzer"
]
