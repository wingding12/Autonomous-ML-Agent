"""
Tests for the Autonomous ML Agent.

This module contains tests for the main agent functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.core.agent import AutonomousMLAgent
from src.config.settings import Settings

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.GEMINI_API_KEY = "test_key"
    settings.E2B_API_KEY = "test_e2b_key"
    return settings

@pytest.fixture
def agent(mock_settings):
    """Create agent instance for testing."""
    return AutonomousMLAgent(mock_settings)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent.agent_id is not None
    assert agent.start_time is not None
    assert not agent.is_initialized

@pytest.mark.asyncio
async def test_agent_initialize(agent):
    """Test agent initialization process."""
    # Mock dependencies
    agent.llm_client = AsyncMock()
    agent.sandbox_manager = AsyncMock()
    agent.data_loader = Mock()
    agent.data_validator = Mock()
    agent.meta_optimizer = Mock()
    agent.evaluator = Mock()
    
    await agent.initialize()
    assert agent.is_initialized

@pytest.mark.asyncio
async def test_agent_cleanup(agent):
    """Test agent cleanup process."""
    # Mock dependencies
    agent.sandbox_manager = AsyncMock()
    agent.llm_client = AsyncMock()
    
    await agent.cleanup()
    assert not agent.is_initialized
