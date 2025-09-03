"""
Configuration models for the Autonomous ML Agent.

This module defines Pydantic models for various configuration options,
including pipeline settings, model parameters, and LLM configurations.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator

class TaskType(str, Enum):
    """Supported machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    BINARY = "binary"

class OptimizationMetric(str, Enum):
    """Supported optimization metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"

class ModelType(str, Enum):
    """Supported model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    KNN = "knn"
    MLP = "mlp"
    SVM = "svm"
    DECISION_TREE = "decision_tree"

class EnsembleStrategy(str, Enum):
    """Supported ensemble strategies."""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAGGING = "bagging"

class HyperparameterSearch(str, Enum):
    """Supported hyperparameter search strategies."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    GRID = "grid"
    HYPERBAND = "hyperband"

class PipelineConfig(BaseModel):
    """Configuration for ML pipeline execution."""
    
    # Basic Configuration
    task_type: TaskType = Field(..., description="Type of ML task")
    target_column: str = Field(..., description="Name of target column")
    optimization_metric: OptimizationMetric = Field(..., description="Metric to optimize")
    
    # Data Configuration
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    cross_validation_folds: int = Field(default=5, ge=3, le=10, description="CV folds")
    
    # Model Configuration
    max_models: int = Field(default=10, ge=1, le=20, description="Maximum models to train")
    enable_ensemble: bool = Field(default=True, description="Enable ensemble methods")
    ensemble_strategies: List[EnsembleStrategy] = Field(
        default=[EnsembleStrategy.VOTING, EnsembleStrategy.STACKING],
        description="Ensemble strategies to try"
    )
    
    # Optimization Configuration
    hyperparameter_search: HyperparameterSearch = Field(
        default=HyperparameterSearch.BAYESIAN,
        description="Hyperparameter search strategy"
    )
    max_trials: int = Field(default=100, ge=10, le=1000, description="Max optimization trials")
    enable_meta_learning: bool = Field(default=True, description="Enable meta-learning warm-starts")
    
    # Runtime Configuration
    max_runtime: int = Field(default=3600, ge=300, le=7200, description="Max runtime in seconds")
    timeout_per_model: int = Field(default=300, ge=60, le=1800, description="Timeout per model")
    
    # Advanced Configuration
    feature_selection: bool = Field(default=True, description="Enable automatic feature selection")
    data_cleaning: bool = Field(default=True, description="Enable automatic data cleaning")
    feature_engineering: bool = Field(default=True, description="Enable feature engineering")
    
    @validator("test_size")
    def validate_test_size(cls, v):
        """Validate test size is reasonable."""
        if v >= 0.5:
            raise ValueError("Test size should be less than 50%")
        return v
    
    @validator("ensemble_strategies")
    def validate_ensemble_strategies(cls, v):
        """Validate ensemble strategies."""
        if not v:
            raise ValueError("At least one ensemble strategy must be specified")
        return v

class ModelConfig(BaseModel):
    """Configuration for individual model training."""
    
    model_type: ModelType = Field(..., description="Type of model to train")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    preprocessing_steps: List[str] = Field(default_factory=list, description="Preprocessing steps")
    
    # Training Configuration
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, ge=1, le=50, description="Early stopping patience")
    
    # Validation Configuration
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3, description="Validation split")
    
    class Config:
        use_enum_values = True

class LLMConfig(BaseModel):
    """Configuration for LLM interactions."""
    
    model_name: str = Field(default="gemini-2.0-flash-exp", description="LLM model name")
    max_tokens: int = Field(default=8192, ge=1000, le=32768, description="Maximum tokens")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    
    # Prompt Configuration
    system_prompt: str = Field(default="", description="System prompt for LLM")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay in seconds")
    
    # Code Generation
    enable_code_execution: bool = Field(default=True, description="Enable code execution")
    code_review: bool = Field(default=True, description="Enable code review by LLM")
    max_code_iterations: int = Field(default=5, ge=1, le=20, description="Max code iterations")
    
    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature value."""
        if v < 0.0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

class SandboxConfig(BaseModel):
    """Configuration for E2B sandbox execution."""
    
    template_id: str = Field(default="base", description="Sandbox template ID")
    timeout: int = Field(default=3600, ge=300, le=7200, description="Sandbox timeout in seconds")
    memory_limit: str = Field(default="4GB", description="Memory limit for sandbox")
    cpu_count: int = Field(default=2, ge=1, le=8, description="CPU count for sandbox")
    
    # File Management
    max_file_size: str = Field(default="100MB", description="Maximum file size")
    allowed_extensions: List[str] = Field(
        default=[".py", ".csv", ".json", ".txt", ".ipynb"],
        description="Allowed file extensions"
    )
    
    # Security
    enable_internet: bool = Field(default=False, description="Enable internet access")
    enable_storage: bool = Field(default=True, description="Enable persistent storage")

class ExperimentConfig(BaseModel):
    """Configuration for experiment tracking."""
    
    experiment_name: str = Field(..., description="Name of the experiment")
    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    artifact_location: Optional[str] = Field(default=None, description="Artifact storage location")
    
    # Logging Configuration
    log_parameters: bool = Field(default=True, description="Log hyperparameters")
    log_metrics: bool = Field(default=True, description="Log metrics")
    log_artifacts: bool = Field(default=True, description="Log model artifacts")
    
    # Versioning
    version_control: bool = Field(default=True, description="Enable version control")
    auto_tag: bool = Field(default=True, description="Auto-tag experiments")
