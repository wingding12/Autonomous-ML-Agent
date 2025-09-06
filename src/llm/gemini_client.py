"""
Google Gemini API client for the Autonomous ML Agent.

This module provides integration with Google's Gemini API, including
code execution capabilities for autonomous ML workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import json

from typing import Any as GenerateContentResponse  # fallback alias when types unavailable

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """
    Client for interacting with Google Gemini API.
    
    This client handles all communication with Gemini, including
    code generation, execution, and analysis tasks.
    """
    
    def __init__(self, settings: Settings):
        """Initialize the Gemini client."""
        self.settings = settings
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.temperature = settings.GEMINI_TEMPERATURE
        
        # Initialize Gemini
        self.enabled: bool = bool(self.api_key)
        if not self.enabled:
            logger.warning("GEMINI_API_KEY not set. LLM features are disabled.")
        self.model = None
        self.is_initialized = False
        
        logger.info(f"Gemini client initialized with model: {self.model_name}")
    
    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        try:
            if not self.enabled:
                raise RuntimeError("Gemini is disabled. Set GEMINI_API_KEY to enable LLM features.")
            # Get the model
            # Lazy import to avoid hard dependency when disabled
            from google import genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                tools=[{"code_execution": {}}]  # Enable code execution
            )
            
            self.is_initialized = True
            logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up the Gemini client."""
        self.model = None
        self.is_initialized = False
        logger.info("Gemini client cleaned up")
    
    async def generate_code(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code using Gemini for a specific ML task.
        
        Args:
            task_description: Description of the code to generate
            context: Additional context (dataset info, requirements, etc.)
            language: Programming language for the code
            
        Returns:
            Dictionary containing generated code and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            # Construct prompt
            prompt = self._build_code_generation_prompt(task_description, context, language)
            
            # Generate content
            response = await self._generate_content(prompt)
            
            # Parse response
            result = self._parse_code_generation_response(response)
            
            logger.info(f"Code generation completed for task: {task_description[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    async def analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dataset characteristics using Gemini.
        
        Args:
            dataset_info: Information about the dataset
            
        Returns:
            Analysis results and recommendations
        """
        if not self.is_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            prompt = self._build_dataset_analysis_prompt(dataset_info)
            response = await self._generate_content(prompt)
            
            result = self._parse_analysis_response(response)
            
            logger.info("Dataset analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            raise
    
    async def optimize_pipeline(
        self,
        current_pipeline: Dict[str, Any],
        performance_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize ML pipeline using Gemini insights.
        
        Args:
            current_pipeline: Current pipeline configuration
            performance_metrics: Current performance metrics
            constraints: Optimization constraints
            
        Returns:
            Optimized pipeline configuration
        """
        if not self.is_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            prompt = self._build_pipeline_optimization_prompt(
                current_pipeline, performance_metrics, constraints
            )
            response = await self._generate_content(prompt)
            
            result = self._parse_optimization_response(response)
            
            logger.info("Pipeline optimization completed")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline optimization failed: {e}")
            raise
    
    async def generate_ensemble_strategy(
        self,
        model_results: List[Dict[str, Any]],
        task_type: str,
        optimization_metric: str
    ) -> Dict[str, Any]:
        """
        Generate ensemble strategy using Gemini.
        
        Args:
            model_results: Results from individual models
            task_type: Type of ML task
            optimization_metric: Metric to optimize
            
        Returns:
            Ensemble strategy configuration
        """
        if not self.is_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            prompt = self._build_ensemble_strategy_prompt(
                model_results, task_type, optimization_metric
            )
            response = await self._generate_content(prompt)
            
            result = self._parse_ensemble_response(response)
            
            logger.info("Ensemble strategy generation completed")
            return result
            
        except Exception as e:
            logger.error(f"Ensemble strategy generation failed: {e}")
            raise
    
    async def _generate_content(self, prompt: str) -> GenerateContentResponse:
        """Generate content using Gemini API."""
        try:
            # Use asyncio to run the synchronous Gemini call
            loop = asyncio.get_event_loop()
            # Lazy import module only when enabled to avoid import errors
            if not self.enabled or not self.model:
                raise RuntimeError("Gemini is disabled or not initialized")
            response = await loop.run_in_executor(
                None,
                self.model.generate_content,
                prompt,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    def _build_code_generation_prompt(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        """Build prompt for code generation."""
        prompt = f"""
        You are an expert machine learning engineer. Generate {language} code for the following task:
        
        TASK: {task_description}
        
        """
        
        if context:
            prompt += f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += """
        REQUIREMENTS:
        - Generate clean, well-documented code
        - Include proper error handling
        - Follow best practices for ML code
        - Use appropriate libraries (scikit-learn, pandas, numpy, etc.)
        - Include comments explaining key decisions
        
        Please provide the complete code implementation.
        """
        
        return prompt
    
    def _build_dataset_analysis_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Build prompt for dataset analysis."""
        prompt = f"""
        You are a data scientist analyzing a machine learning dataset. 
        Please analyze the following dataset information and provide insights:
        
        DATASET INFO:
        {json.dumps(dataset_info, indent=2)}
        
        Please provide:
        1. Key characteristics of the dataset
        2. Potential challenges and issues
        3. Recommended preprocessing steps
        4. Suitable ML algorithms
        5. Feature engineering suggestions
        6. Validation strategy recommendations
        
        Be specific and actionable in your recommendations.
        """
        
        return prompt
    
    def _build_pipeline_optimization_prompt(
        self,
        current_pipeline: Dict[str, Any],
        performance_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for pipeline optimization."""
        prompt = f"""
        You are an ML engineer optimizing a machine learning pipeline.
        Please analyze the current pipeline and suggest improvements:
        
        CURRENT PIPELINE:
        {json.dumps(current_pipeline, indent=2)}
        
        PERFORMANCE METRICS:
        {json.dumps(performance_metrics, indent=2)}
        
        """
        
        if constraints:
            prompt += f"CONSTRAINTS:\n{json.dumps(constraints, indent=2)}\n\n"
        
        prompt += """
        Please provide:
        1. Analysis of current performance
        2. Specific optimization recommendations
        3. Hyperparameter tuning suggestions
        4. Feature engineering improvements
        5. Model architecture changes
        6. Expected performance improvements
        
        Be specific and prioritize the most impactful changes.
        """
        
        return prompt
    
    def _build_ensemble_strategy_prompt(
        self,
        model_results: List[Dict[str, Any]],
        task_type: str,
        optimization_metric: str
    ) -> str:
        """Build prompt for ensemble strategy generation."""
        prompt = f"""
        You are an ML engineer designing ensemble strategies.
        Please analyze the following model results and suggest ensemble approaches:
        
        MODEL RESULTS:
        {json.dumps(model_results, indent=2)}
        
        TASK TYPE: {task_type}
        OPTIMIZATION METRIC: {optimization_metric}
        
        Please provide:
        1. Analysis of individual model performance
        2. Recommended ensemble strategy (voting, stacking, blending)
        3. Specific implementation details
        4. Expected performance improvements
        5. Trade-offs and considerations
        
        Focus on practical, implementable ensemble strategies.
        """
        
        return prompt
    
    def _parse_code_generation_response(self, response: GenerateContentResponse) -> Dict[str, Any]:
        """Parse code generation response."""
        try:
            # Extract text content
            text_content = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
            
            # Extract executable code if available
            executable_code = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'executable_code') and part.executable_code:
                    executable_code = part.executable_code.code
            
            return {
                "code": executable_code or text_content,
                "text": text_content,
                "has_executable_code": executable_code is not None,
                "model": self.model_name,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse code generation response: {e}")
            return {
                "code": "",
                "text": str(response),
                "has_executable_code": False,
                "error": str(e)
            }
    
    def _parse_analysis_response(self, response: GenerateContentResponse) -> Dict[str, Any]:
        """Parse analysis response."""
        try:
            text_content = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
            
            return {
                "analysis": text_content,
                "model": self.model_name,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return {"error": str(e)}
    
    def _parse_optimization_response(self, response: GenerateContentResponse) -> Dict[str, Any]:
        """Parse optimization response."""
        try:
            text_content = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
            
            return {
                "optimization_plan": text_content,
                "model": self.model_name,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse optimization response: {e}")
            return {"error": str(e)}
    
    def _parse_ensemble_response(self, response: GenerateContentResponse) -> Dict[str, Any]:
        """Parse ensemble strategy response."""
        try:
            text_content = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
            
            return {
                "ensemble_strategy": text_content,
                "model": self.model_name,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse ensemble response: {e}")
            return {"error": str(e)}
