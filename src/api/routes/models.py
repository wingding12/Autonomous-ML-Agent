"""
Models API routes for the Autonomous ML Agent.

This module provides REST API endpoints for model management,
leaderboard access, and model details.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ...core.agent import AutonomousMLAgent
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global agent reference (will be set by main.py)
agent: Optional[AutonomousMLAgent] = None

def set_agent(agent_instance: AutonomousMLAgent):
    """Set the global agent reference."""
    global agent
    agent = agent_instance

class ModelInfo(BaseModel):
    """Model information model."""
    model_id: str
    model_name: str
    model_type: str
    pipeline_id: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    created_at: str
    status: str

class LeaderboardResponse(BaseModel):
    """Leaderboard response model."""
    models: List[ModelInfo]
    total_models: int
    best_model: Optional[ModelInfo] = None
    average_performance: Dict[str, float]

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    pipeline_id: Optional[str] = None,
    limit: int = 50,
    sort_by: str = "accuracy"
):
    """
    Get the model leaderboard.
    
    Args:
        pipeline_id: Optional pipeline ID to filter models
        limit: Maximum number of models to return
        sort_by: Metric to sort by (accuracy, precision, recall, f1_score, etc.)
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual leaderboard logic
        # For now, return placeholder data
        models = []
        
        # Mock data for demonstration
        if pipeline_id:
            # Filter by pipeline ID
            pass
        
        # Sort models by specified metric
        # models.sort(key=lambda x: x.performance_metrics.get(sort_by, 0), reverse=True)
        
        # Limit results
        models = models[:limit]
        
        # Calculate statistics
        total_models = len(models)
        best_model = models[0] if models else None
        
        # Calculate average performance
        avg_performance = {}
        if models:
            metrics = models[0].performance_metrics.keys()
            for metric in metrics:
                values = [m.performance_metrics.get(metric, 0) for m in models]
                avg_performance[metric] = sum(values) / len(values)
        
        return LeaderboardResponse(
            models=models,
            total_models=total_models,
            best_model=best_model,
            average_performance=avg_performance
        )
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )

@router.get("/{model_id}/details", response_model=ModelInfo)
async def get_model_details(model_id: str):
    """Get detailed information about a specific model."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual model retrieval logic
        # For now, return placeholder data
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Model details endpoint not yet implemented"
        )
        
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model details: {str(e)}"
        )

@router.get("/", response_model=List[ModelInfo])
async def list_models(
    pipeline_id: Optional[str] = None,
    model_type: Optional[str] = None,
    status: Optional[str] = None
):
    """List all models with optional filtering."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual model listing logic
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )
