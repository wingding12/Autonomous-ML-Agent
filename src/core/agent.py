"""
Autonomous ML Agent - Main orchestration class.

This module contains the core AutonomousMLAgent class that coordinates
all aspects of the machine learning pipeline, from data ingestion to
model deployment.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..config.settings import Settings
from ..config.models import PipelineConfig, TaskType, OptimizationMetric
from ..llm.gemini_client import GeminiClient
from ..sandbox.e2b_manager import E2BManager
from ..data.loader import DataLoader
from ..data.validator import DataValidator
from ..ml.optimization.meta_learning import MetaLearningOptimizer
from ..ml.evaluation.evaluator import ModelEvaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AutonomousMLAgent:
    """
    Autonomous Machine Learning Agent that orchestrates end-to-end ML pipelines.
    
    This agent uses LLMs to make intelligent decisions about data preprocessing,
    model selection, hyperparameter optimization, and ensemble strategies.
    """
    
    def __init__(self, settings: Settings):
        """Initialize the autonomous ML agent."""
        self.settings = settings
        self.agent_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Core components
        self.llm_client: Optional[GeminiClient] = None
        self.sandbox_manager: Optional[E2BManager] = None
        self.data_loader: Optional[DataLoader] = None
        self.data_validator: Optional[DataValidator] = None
        self.meta_optimizer: Optional[MetaLearningOptimizer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        
        # State management
        self.active_pipelines: Dict[str, Any] = {}
        self.completed_pipelines: Dict[str, Any] = {}
        self.failed_pipelines: Dict[str, Any] = {}
        self.is_initialized = False
        
        logger.info(f"Autonomous ML Agent initialized with ID: {self.agent_id}")
    
    async def initialize(self) -> None:
        """Initialize all agent components."""
        try:
            logger.info("Initializing Autonomous ML Agent components...")
            
            # Initialize LLM client
            self.llm_client = GeminiClient(self.settings)
            await self.llm_client.initialize()
            
            # Initialize sandbox manager
            self.sandbox_manager = E2BManager(self.settings)
            await self.sandbox_manager.initialize()
            
            # Initialize data components
            self.data_loader = DataLoader()
            self.data_validator = DataValidator()
            
            # Initialize ML components
            self.meta_optimizer = MetaLearningOptimizer(self.settings)
            self.evaluator = ModelEvaluator()
            
            self.is_initialized = True
            logger.info("Autonomous ML Agent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        try:
            logger.info("Cleaning up Autonomous ML Agent...")
            
            # Clean up active pipelines
            for pipeline_id in list(self.active_pipelines.keys()):
                await self.stop_pipeline(pipeline_id)
            
            # Clean up sandbox manager
            if self.sandbox_manager:
                await self.sandbox_manager.cleanup()
            
            # Clean up LLM client
            if self.llm_client:
                await self.llm_client.cleanup()
            
            logger.info("Autonomous ML Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def run_pipeline(
        self,
        dataset_path: Union[str, Path],
        target_column: str,
        task_type: Union[TaskType, str],
        optimization_metric: Union[OptimizationMetric, str],
        pipeline_config: Optional[PipelineConfig] = None,
        **kwargs
    ) -> str:
        """
        Run a complete ML pipeline autonomously.
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Name of the target column
            task_type: Type of ML task (classification, regression, etc.)
            optimization_metric: Metric to optimize
            pipeline_config: Optional custom pipeline configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Pipeline ID for tracking
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # Generate pipeline ID
        pipeline_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting pipeline {pipeline_id} for dataset: {dataset_path}")
            
            # Create pipeline configuration
            if pipeline_config is None:
                pipeline_config = self._create_default_pipeline_config(
                    task_type, optimization_metric, **kwargs
                )
            
            # Initialize pipeline
            pipeline = await self._create_pipeline(
                pipeline_id, dataset_path, target_column, pipeline_config
            )
            
            # Add to active pipelines
            self.active_pipelines[pipeline_id] = pipeline
            
            # Start pipeline execution asynchronously
            asyncio.create_task(self._execute_pipeline(pipeline_id))
            
            logger.info(f"Pipeline {pipeline_id} started successfully")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to start pipeline {pipeline_id}: {e}")
            self.failed_pipelines[pipeline_id] = {
                "error": str(e),
                "timestamp": datetime.now(),
                "status": "failed"
            }
            raise
    
    async def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the current status of a pipeline."""
        if pipeline_id in self.active_pipelines:
            pipeline = self.active_pipelines[pipeline_id]
            return await pipeline.get_status()
        elif pipeline_id in self.completed_pipelines:
            return self.completed_pipelines[pipeline_id]
        elif pipeline_id in self.failed_pipelines:
            return self.failed_pipelines[pipeline_id]
        else:
            raise ValueError(f"Pipeline {pipeline_id} not found")
    
    async def get_pipeline_results(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the results of a completed pipeline."""
        if pipeline_id in self.completed_pipelines:
            return self.completed_pipelines[pipeline_id]
        elif pipeline_id in self.failed_pipelines:
            raise RuntimeError(f"Pipeline {pipeline_id} failed: {self.failed_pipelines[pipeline_id]['error']}")
        else:
            raise ValueError(f"Pipeline {pipeline_id} not found or not completed")
    
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop a running pipeline."""
        if pipeline_id in self.active_pipelines:
            pipeline = self.active_pipelines[pipeline_id]
            await pipeline.stop()
            del self.active_pipelines[pipeline_id]
            logger.info(f"Pipeline {pipeline_id} stopped")
            return True
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get overall agent status."""
        return {
            "agent_id": self.agent_id,
            "status": "running" if self.is_initialized else "initializing",
            "start_time": self.start_time.isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "active_pipelines": len(self.active_pipelines),
            "completed_pipelines": len(self.completed_pipelines),
            "failed_pipelines": len(self.failed_pipelines),
            "total_pipelines": len(self.active_pipelines) + len(self.completed_pipelines) + len(self.failed_pipelines)
        }
    
    def _create_default_pipeline_config(
        self,
        task_type: Union[TaskType, str],
        optimization_metric: Union[OptimizationMetric, str],
        **kwargs
    ) -> PipelineConfig:
        """Create default pipeline configuration."""
        # Convert string enums if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if isinstance(optimization_metric, str):
            optimization_metric = OptimizationMetric(optimization_metric)
        
        # Create base config
        config_data = {
            "task_type": task_type,
            "optimization_metric": optimization_metric,
            **kwargs
        }
        
        return PipelineConfig(**config_data)
    
    async def _create_pipeline(
        self,
        pipeline_id: str,
        dataset_path: Union[str, Path],
        target_column: str,
        config: PipelineConfig
    ) -> Any:
        """Create a new pipeline instance."""
        # This would create a pipeline object
        # For now, return a placeholder
        return {
            "id": pipeline_id,
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "config": config,
            "status": "created",
            "created_at": datetime.now()
        }
    
    async def _execute_pipeline(self, pipeline_id: str) -> None:
        """Execute a pipeline asynchronously."""
        try:
            pipeline = self.active_pipelines[pipeline_id]
            logger.info(f"Executing pipeline {pipeline_id}")
            
            # Simulate pipeline execution
            await asyncio.sleep(5)  # Placeholder
            
            # Move to completed
            pipeline["status"] = "completed"
            pipeline["completed_at"] = datetime.now()
            self.completed_pipelines[pipeline_id] = pipeline
            del self.active_pipelines[pipeline_id]
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} execution failed: {e}")
            pipeline["status"] = "failed"
            pipeline["error"] = str(e)
            pipeline["failed_at"] = datetime.now()
            self.failed_pipelines[pipeline_id] = pipeline
            del self.active_pipelines[pipeline_id]
    
    async def analyze_dataset(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a dataset to understand its characteristics."""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        
        try:
            # Load and validate dataset
            data = await self.data_loader.load_dataset(dataset_path)
            analysis = await self.data_validator.analyze_dataset(data)
            
            # Use LLM to generate insights
            insights = await self.llm_client.analyze_dataset(analysis)
            
            return {
                "dataset_info": analysis,
                "llm_insights": insights,
                "recommendations": await self._generate_recommendations(analysis)
            }
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            raise
    
    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dataset analysis."""
        # This would use the LLM to generate specific recommendations
        # For now, return placeholder recommendations
        return [
            "Consider feature scaling for numerical features",
            "Handle missing values appropriately",
            "Use cross-validation for robust evaluation"
        ]
