"""
Sandbox package for the Autonomous ML Agent.

This package handles secure code execution in isolated environments
using E2B sandboxes for ML workflows.
"""

from .e2b_manager import E2BManager
from .code_runner import CodeRunner
from .environment import SandboxEnvironment

__all__ = [
    "E2BManager",
    "CodeRunner",
    "SandboxEnvironment"
]
