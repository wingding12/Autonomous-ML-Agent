"""
API package for the Autonomous ML Agent.

This package contains all FastAPI routes and middleware for the
REST API service.
"""

from .routes import pipeline, models, inference
from .middleware import setup_middleware

__all__ = [
    "pipeline",
    "models", 
    "inference",
    "setup_middleware"
]
