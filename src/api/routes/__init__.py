"""
API routes package for the Autonomous ML Agent.

This package contains all FastAPI route definitions organized by
functionality (pipeline, models, inference).
"""

from . import pipeline
from . import models
from . import inference

__all__ = [
    "pipeline",
    "models",
    "inference"
]
