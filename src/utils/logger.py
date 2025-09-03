"""
Logging utilities for the Autonomous ML Agent.

This module provides centralized logging configuration and utilities
for consistent logging across the application.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    config_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, text)
        log_file: Optional log file path
        config_file: Optional logging config file path
    """
    if config_file and Path(config_file).exists():
        # Load logging configuration from file
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        return
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    if log_format.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("e2b").setLevel(logging.INFO)
    logging.getLogger("google.generativeai").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class StructuredLogger:
    """Structured logger with additional context support."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.extra_fields: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind additional context to the logger."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.extra_fields = {**self.extra_fields, **kwargs}
        return new_logger
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with extra fields."""
        extra = {**self.extra_fields, **kwargs}
        self.logger.log(level, message, extra={"extra_fields": extra})
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception message with traceback."""
        extra = {**self.extra_fields, **kwargs}
        self.logger.exception(message, extra={"extra_fields": extra})

def setup_mlflow_logging(experiment_name: str, run_name: str) -> None:
    """
    Setup MLflow logging for experiment tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the current run
    """
    try:
        import mlflow
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        mlflow.start_run(run_name=run_name)
        
        # Log system info
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("platform", sys.platform)
        
        logging.info(f"MLflow logging setup for experiment: {experiment_name}, run: {run_name}")
        
    except ImportError:
        logging.warning("MLflow not available, skipping MLflow logging setup")
    except Exception as e:
        logging.error(f"Failed to setup MLflow logging: {e}")

def log_pipeline_event(
    pipeline_id: str,
    event_type: str,
    details: Dict[str, Any],
    logger_name: str = "pipeline"
) -> None:
    """
    Log pipeline events with structured information.
    
    Args:
        pipeline_id: ID of the pipeline
        event_type: Type of event (started, completed, failed, etc.)
        details: Additional event details
        logger_name: Name of the logger to use
    """
    logger = get_logger(logger_name)
    
    log_data = {
        "pipeline_id": pipeline_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **details
    }
    
    logger.info(f"Pipeline event: {event_type}", extra={"extra_fields": log_data})

def log_model_training(
    model_name: str,
    dataset_info: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    logger_name: str = "model_training"
) -> None:
    """
    Log model training information.
    
    Args:
        model_name: Name of the model
        dataset_info: Information about the dataset
        hyperparameters: Model hyperparameters
        metrics: Training metrics
        logger_name: Name of the logger to use
    """
    logger = get_logger(logger_name)
    
    log_data = {
        "model_name": model_name,
        "dataset_info": dataset_info,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Model training completed: {model_name}", extra={"extra_fields": log_data})
