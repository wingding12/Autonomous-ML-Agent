"""
Autonomous ML Agent - An intelligent, end-to-end machine learning pipeline orchestrated by LLMs.

This package provides a comprehensive framework for autonomous machine learning workflows,
including data preprocessing, model training, optimization, and deployment.
"""

__version__ = "0.1.0"
__author__ = "Autonomous ML Team"
__email__ = "team@autonomousml.com"

from .core.agent import AutonomousMLAgent
from .core.pipeline import MLPipeline
from .core.executor import PipelineExecutor

__all__ = [
    "AutonomousMLAgent",
    "MLPipeline", 
    "PipelineExecutor",
]
