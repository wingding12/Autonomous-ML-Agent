"""
Inference API routes for the Autonomous ML Agent.

This module provides REST API endpoints for model inference,
including single prediction and batch processing.
"""

from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel
import json

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

class SinglePredictionRequest(BaseModel):
    """Request model for single prediction."""
    model_id: str
    features: Dict[str, Union[str, int, float]]
    return_probabilities: Optional[bool] = False
    return_confidence: Optional[bool] = False

class SinglePredictionResponse(BaseModel):
    """Response model for single prediction."""
    prediction: Union[str, int, float]
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    model_id: str
    prediction_time: float
    feature_importance: Optional[Dict[str, float]] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    model_id: str
    data_file: UploadFile
    return_probabilities: Optional[bool] = False
    return_confidence: Optional[bool] = False
    return_feature_importance: Optional[bool] = False

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[Union[str, int, float]]
    probabilities: Optional[List[Dict[str, float]]] = None
    confidence: Optional[List[float]] = None
    model_id: str
    total_predictions: int
    processing_time: float
    feature_importance: Optional[Dict[str, float]] = None

@router.post("/single", response_model=SinglePredictionResponse)
async def single_prediction(request: SinglePredictionRequest):
    """
    Make a single prediction using a trained model.
    
    Args:
        request: Prediction request with features and model ID
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual single prediction logic
        # For now, return placeholder data
        
        # Mock prediction
        prediction = "class_1"  # Placeholder
        probabilities = {"class_1": 0.8, "class_2": 0.2} if request.return_probabilities else None
        confidence = 0.85 if request.return_confidence else None
        
        return SinglePredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            model_id=request.model_id,
            prediction_time=0.05,  # Mock time
            feature_importance=None
        )
        
    except Exception as e:
        logger.error(f"Single prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_prediction(
    model_id: str = File(...),
    data_file: UploadFile = File(...),
    return_probabilities: bool = False,
    return_confidence: bool = False,
    return_feature_importance: bool = False
):
    """
    Make batch predictions using a trained model.
    
    Args:
        model_id: ID of the model to use for prediction
        data_file: CSV file containing features for prediction
        return_probabilities: Whether to return prediction probabilities
        return_confidence: Whether to return confidence scores
        return_feature_importance: Whether to return feature importance
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Validate file type
        if not data_file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data file must be a CSV file"
            )
        
        # This would implement actual batch prediction logic
        # For now, return placeholder data
        
        # Mock batch predictions
        predictions = ["class_1", "class_2", "class_1"]  # Placeholder
        probabilities = None
        if return_probabilities:
            probabilities = [
                {"class_1": 0.8, "class_2": 0.2},
                {"class_1": 0.3, "class_2": 0.7},
                {"class_1": 0.9, "class_2": 0.1}
            ]
        
        confidence = None
        if return_confidence:
            confidence = [0.85, 0.78, 0.92]
        
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            model_id=model_id,
            total_predictions=len(predictions),
            processing_time=0.15,  # Mock time
            feature_importance=None
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/models/{model_id}/capabilities")
async def get_model_capabilities(model_id: str):
    """Get the capabilities and metadata of a specific model."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual model capability retrieval
        # For now, return placeholder data
        
        capabilities = {
            "model_id": model_id,
            "model_type": "random_forest",
            "task_type": "classification",
            "supports_probabilities": True,
            "supports_confidence": True,
            "supports_feature_importance": True,
            "input_features": ["feature1", "feature2", "feature3"],
            "output_classes": ["class_1", "class_2"],
            "model_size": "2.5MB",
            "training_date": "2024-01-15T10:30:00Z"
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get model capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model capabilities: {str(e)}"
        )

@router.post("/models/{model_id}/validate")
async def validate_model_input(
    model_id: str,
    features: Dict[str, Union[str, int, float]]
):
    """
    Validate input features for a specific model.
    
    Args:
        model_id: ID of the model to validate against
        features: Features to validate
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement actual feature validation logic
        # For now, return placeholder validation result
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "feature_info": {
                "feature1": {"type": "numeric", "range": [0, 100]},
                "feature2": {"type": "categorical", "values": ["A", "B", "C"]},
                "feature3": {"type": "numeric", "range": [-1, 1]}
            }
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Feature validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature validation failed: {str(e)}"
        )

@router.get("/health")
async def inference_health_check():
    """Health check for the inference service."""
    return {
        "status": "healthy",
        "service": "inference",
        "timestamp": "2024-01-15T10:30:00Z"
    }
