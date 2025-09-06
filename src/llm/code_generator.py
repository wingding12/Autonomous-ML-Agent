"""
Code Generator for the Autonomous ML Agent.

This module provides code generation functionality
using LLM capabilities.
"""

from typing import Dict, Any, Optional
import logging

from config.settings import settings
from .gemini_client import GeminiClient
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Code generation component."""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize the code generator."""
        self.logger = logger
        self.client = client or GeminiClient(settings)
        self.prompts = PromptTemplates()
    
    async def generate_preprocessing_code(self, dataset_info: Dict[str, Any]) -> str:
        """Generate preprocessing code using LLM, fallback to template if disabled."""
        if not self.client.enabled:
            return "\n".join([
                "# Fallback preprocessing pipeline",
                "import pandas as pd",
                "from sklearn.model_selection import train_test_split",
                "from sklearn.compose import ColumnTransformer",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
                "",
                "def build_preprocess(X):",
                "    num_cols = X.select_dtypes(include=['number']).columns.tolist()",
                "    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()",
                "    pre = ColumnTransformer([",
                "        ('num', StandardScaler(), num_cols),",
                "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)",
                "    ])",
                "    return pre",
            ])
        prompt = self.prompts.get_code_generation_prompt(
            "Create a robust preprocessing pipeline for tabular data (missing values, categorical encoding, scaling).",
            {"dataset_info": dataset_info}
        )
        await self.client.initialize()
        result = await self.client.generate_code(prompt)
        return result.get("code", "")
    
    async def generate_model_code(self, model_type: str, hyperparameters: Dict[str, Any]) -> str:
        """Generate model training code using LLM, with fallback template."""
        if not self.client.enabled:
            return f"# Fallback training for {model_type}\n# hyperparameters: {hyperparameters}"
        task = f"Train a {model_type} with given hyperparameters and scikit-learn API."
        prompt = self.prompts.get_code_generation_prompt(task, {"hyperparameters": hyperparameters})
        await self.client.initialize()
        result = await self.client.generate_code(prompt)
        return result.get("code", "")
    
    async def generate_evaluation_code(self, task_type: str) -> str:
        """Generate evaluation code using LLM, with fallback template."""
        if not self.client.enabled:
            return "\n".join([
                "# Fallback evaluation",
                "from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score",
                "",
                "def evaluate(task, y_true, y_pred):",
                "    if task in ('classification', 'binary', 'multiclass'):
        return {'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, average='macro')}",
                "    else:
        return {'mse': mean_squared_error(y_true, y_pred), 'r2': r2_score(y_true, y_pred)}",
            ])
        task = f"Write evaluation utilities for {task_type} tasks."
        prompt = self.prompts.get_code_generation_prompt(task)
        await self.client.initialize()
        result = await self.client.generate_code(prompt)
        return result.get("code", "")
