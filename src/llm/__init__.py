"""
LLM Integration package for the Autonomous ML Agent.

This package handles all interactions with Large Language Models,
including code generation, analysis, and decision making.
"""

from .gemini_client import GeminiClient
from .prompt_templates import PromptTemplates
from .code_generator import CodeGenerator

__all__ = [
    "GeminiClient",
    "PromptTemplates",
    "CodeGenerator"
]
