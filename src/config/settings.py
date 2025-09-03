"""
Application settings and configuration management.

This module handles all environment variables and application configuration,
providing a centralized way to manage settings across the application.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM Configuration
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-exp", env="GEMINI_MODEL")
    GEMINI_MAX_TOKENS: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    GEMINI_TEMPERATURE: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    
    # E2B Sandbox Configuration
    E2B_API_KEY: str = Field(..., env="E2B_API_KEY")
    E2B_TEMPLATE_ID: str = Field(default="base", env="E2B_TEMPLATE_ID")
    E2B_SANDBOX_TIMEOUT: int = Field(default=3600, env="E2B_SANDBOX_TIMEOUT")
    
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./autonomous_ml.db", env="DATABASE_URL")
    
    # Cache Configuration
    MODEL_CACHE_DIR: str = Field(default="./cache/models", env="MODEL_CACHE_DIR")
    EXPERIMENT_CACHE_DIR: str = Field(default="./cache/experiments", env="EXPERIMENT_CACHE_DIR")
    
    # Pipeline Configuration
    MAX_PIPELINE_RUNTIME: int = Field(default=3600, env="MAX_PIPELINE_RUNTIME")
    MAX_MODELS_PER_PIPELINE: int = Field(default=10, env="MAX_MODELS_PER_PIPELINE")
    DEFAULT_CV_FOLDS: int = Field(default=5, env="DEFAULT_CV_FOLDS")
    ENABLE_META_LEARNING: bool = Field(default=True, env="ENABLE_META_LEARNING")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="autonomous_ml", env="MLFLOW_EXPERIMENT_NAME")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string to list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("GEMINI_TEMPERATURE")
    def validate_temperature(cls, v):
        """Validate temperature value."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("GEMINI_TEMPERATURE must be between 0.0 and 2.0")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Export settings for easy access
settings = get_settings()
