"""
Pytest configuration for the Autonomous ML Agent.

This module provides pytest fixtures and configuration
for testing the application.
"""

import pytest
import asyncio
from typing import Dict, Any

@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Sample pipeline configuration for testing."""
    return {
        "task_type": "classification",
        "target_column": "target",
        "optimization_metric": "accuracy",
        "test_size": 0.2,
        "random_state": 42,
        "cross_validation_folds": 5,
        "max_models": 10,
        "enable_ensemble": True,
        "max_runtime": 3600,
        "enable_meta_learning": True
    }

@pytest.fixture
def sample_dataset_info() -> Dict[str, Any]:
    """Sample dataset information for testing."""
    return {
        "shape": (100, 10),
        "dtypes": {"feature1": "float64", "feature2": "object"},
        "missing_values": {"feature1": 0, "feature2": 5},
        "unique_values": {"feature1": 95, "feature2": 3}
    }

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
