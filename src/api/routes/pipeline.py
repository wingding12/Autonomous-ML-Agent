"""
Pipeline API routes for the Autonomous ML Agent.

This module provides REST API endpoints for creating, managing,
and monitoring ML pipelines.
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...core.agent import AutonomousMLAgent
from ...config.models import PipelineConfig, TaskType, OptimizationMetric
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global agent reference (will be set by main.py)
agent: Optional[AutonomousMLAgent] = None

def set_agent(agent_instance: AutonomousMLAgent):
    """Set the global agent reference."""
    global agent
    agent = agent_instance

class PipelineCreateRequest(BaseModel):
    """Request model for creating a new pipeline."""
    target_column: str
    task_type: TaskType
    optimization_metric: OptimizationMetric
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    cross_validation_folds: Optional[int] = 5
    max_models: Optional[int] = 10
    enable_ensemble: Optional[bool] = True
    max_runtime: Optional[int] = 3600
    enable_meta_learning: Optional[bool] = True

class PipelineResponse(BaseModel):
    """Response model for pipeline operations."""
    pipeline_id: str
    status: str
    message: str
    created_at: str
    estimated_completion: Optional[str] = None

class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""
    pipeline_id: str
    status: str
    progress: Optional[float] = None
    current_step: Optional[str] = None
    estimated_completion: Optional[str] = None
    created_at: str
    updated_at: str

class PipelineResultsResponse(BaseModel):
    """Response model for pipeline results."""
    pipeline_id: str
    status: str
    best_model: Dict[str, Any]
    all_models: List[Dict[str, Any]]
    ensemble_strategy: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float]
    preprocessing_steps: List[str]
    training_time: float
    completed_at: str

@router.post("/create", response_model=PipelineResponse)
async def create_pipeline(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: TaskType = Form(...),
    optimization_metric: OptimizationMetric = Form(...),
    test_size: Optional[float] = Form(0.2),
    random_state: Optional[int] = Form(42),
    cross_validation_folds: Optional[int] = Form(5),
    max_models: Optional[int] = Form(10),
    enable_ensemble: Optional[bool] = Form(True),
    max_runtime: Optional[int] = Form(3600),
    enable_meta_learning: Optional[bool] = Form(True)
):
    """
    Create a new ML pipeline.
    
    This endpoint accepts a dataset file and configuration parameters,
    then creates and starts an autonomous ML pipeline.
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Validate file type
        if not dataset.filename.lower().endswith(('.csv', '.xlsx', '.parquet')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Supported formats: CSV, Excel, Parquet"
            )
        
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{dataset.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await dataset.read()
            buffer.write(content)
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            task_type=task_type,
            target_column=target_column,
            optimization_metric=optimization_metric,
            test_size=test_size,
            random_state=random_state,
            cross_validation_folds=cross_validation_folds,
            max_models=max_models,
            enable_ensemble=enable_ensemble,
            max_runtime=max_runtime,
            enable_meta_learning=enable_meta_learning
        )
        
        # Start pipeline
        pipeline_id = await agent.run_pipeline(
            dataset_path=temp_file_path,
            target_column=target_column,
            task_type=task_type,
            optimization_metric=optimization_metric,
            pipeline_config=pipeline_config
        )
        
        logger.info(f"Pipeline {pipeline_id} created successfully")
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="created",
            message="Pipeline created and started successfully",
            created_at=asyncio.get_event_loop().time()
        )
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pipeline: {str(e)}"
        )

@router.get("/{pipeline_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str):
    """Get the current status of a pipeline."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        status_info = await agent.get_pipeline_status(pipeline_id)
        
        return PipelineStatusResponse(
            pipeline_id=pipeline_id,
            status=status_info.get("status", "unknown"),
            progress=status_info.get("progress"),
            current_step=status_info.get("current_step"),
            estimated_completion=status_info.get("estimated_completion"),
            created_at=status_info.get("created_at", ""),
            updated_at=asyncio.get_event_loop().time()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}"
        )

@router.get("/{pipeline_id}/results", response_model=PipelineResultsResponse)
async def get_pipeline_results(pipeline_id: str):
    """Get the results of a completed pipeline."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        results = await agent.get_pipeline_results(pipeline_id)
        
        return PipelineResultsResponse(
            pipeline_id=pipeline_id,
            status=results.get("status", "completed"),
            best_model=results.get("best_model", {}),
            all_models=results.get("all_models", []),
            ensemble_strategy=results.get("ensemble_strategy"),
            feature_importance=results.get("feature_importance"),
            performance_metrics=results.get("performance_metrics", {}),
            preprocessing_steps=results.get("preprocessing_steps", []),
            training_time=results.get("training_time", 0.0),
            completed_at=results.get("completed_at", "")
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline not completed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline results: {str(e)}"
        )

@router.delete("/{pipeline_id}")
async def stop_pipeline(pipeline_id: str):
    """Stop a running pipeline."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        success = await agent.stop_pipeline(pipeline_id)
        
        if success:
            return JSONResponse(
                content={"message": f"Pipeline {pipeline_id} stopped successfully"},
                status_code=status.HTTP_200_OK
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline {pipeline_id} not found or already stopped"
            )
            
    except Exception as e:
        logger.error(f"Failed to stop pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop pipeline: {str(e)}"
        )

@router.get("/", response_model=List[Dict[str, Any]])
async def list_pipelines():
    """List all pipelines with their status."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Get agent status which includes pipeline counts
        agent_status = await agent.get_status()
        
        pipelines = []
        
        # Add active pipelines
        for pipeline_id in agent.active_pipelines.keys():
            try:
                status_info = await agent.get_pipeline_status(pipeline_id)
                pipelines.append({
                    "pipeline_id": pipeline_id,
                    "status": "active",
                    **status_info
                })
            except Exception as e:
                logger.warning(f"Failed to get status for pipeline {pipeline_id}: {e}")
        
        # Add completed pipelines
        for pipeline_id, pipeline_info in agent.completed_pipelines.items():
            pipelines.append({
                "pipeline_id": pipeline_id,
                "status": "completed",
                **pipeline_info
            })
        
        # Add failed pipelines
        for pipeline_id, pipeline_info in agent.failed_pipelines.items():
            pipelines.append({
                "pipeline_id": pipeline_id,
                "status": "failed",
                **pipeline_info
            })
        
        return pipelines
        
    except Exception as e:
        logger.error(f"Failed to list pipelines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pipelines: {str(e)}"
        )

@router.post("/{pipeline_id}/export")
async def export_pipeline(pipeline_id: str):
    """Export a completed pipeline as a deployable artifact."""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # This would implement pipeline export functionality
        # For now, return a placeholder response
        return JSONResponse(
            content={
                "message": f"Pipeline {pipeline_id} export initiated",
                "export_format": "python_package",
                "estimated_time": "2-5 minutes"
            },
            status_code=status.HTTP_202_ACCEPTED
        )
        
    except Exception as e:
        logger.error(f"Failed to export pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export pipeline: {str(e)}"
        )
