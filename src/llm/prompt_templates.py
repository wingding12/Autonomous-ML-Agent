"""
Prompt Templates for the Autonomous ML Agent.

This module provides prompt templates for various
LLM interactions and tasks.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptTemplates:
    """Prompt template management component."""
    
    def __init__(self):
        """Initialize the prompt templates."""
        self.logger = logger
    
    def get_code_generation_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get structured prompt for code generation."""
        base = [
            "You are a senior ML engineer writing production-grade code.",
            f"TASK: {task}",
            "REQUIREMENTS:",
            "- Robust error handling",
            "- Clear, modular functions",
            "- Scikit-learn, pandas, numpy where appropriate",
            "- Return values and brief summary",
        ]
        if context:
            base.append("CONTEXT:\n" + json_dumps(context))
        return "\n".join(base)
    
    def get_analysis_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Get structured prompt for dataset analysis."""
        return (
            "You are a data scientist. Analyze the dataset info provided and give: "
            "key risks, cleaning steps, encoding/scaling advice, validation plan, and model shortlist.\n"
            f"DATASET_INFO:\n{json_dumps(dataset_info)}"
        )
    
    def get_optimization_prompt(self, pipeline_info: Dict[str, Any]) -> str:
        """Get structured prompt for pipeline optimization."""
        return (
            "You are optimizing an ML pipeline under time/compute constraints."
            " Provide concrete hyperparameter ranges and trial budget splits by model.\n"
            f"PIPELINE_INFO:\n{json_dumps(pipeline_info)}"
        )


def json_dumps(data: Dict[str, Any]) -> str:
    import json
    return json.dumps(data, indent=2, default=str)
