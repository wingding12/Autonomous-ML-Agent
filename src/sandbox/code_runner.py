"""
Code Runner for the Autonomous ML Agent.

This module provides code execution functionality in
E2B sandboxes.
"""

from typing import Dict, Any, Optional
import logging

from config.settings import settings
from sandbox.e2b_manager import E2BManager

logger = logging.getLogger(__name__)

class CodeRunner:
    """Code execution component using E2BManager."""
    
    def __init__(self, manager: Optional[E2BManager] = None):
        """Initialize the code runner."""
        self.logger = logger
        self.manager = manager or E2BManager(settings)
    
    async def run_code(self, code: str, sandbox_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Python code in the specified sandbox.
        
        Args:
            code: Python code to execute
            sandbox_id: Target sandbox identifier
            timeout: Optional execution timeout
        Returns:
            Execution result dict
        """
        if not self.manager.enabled:
            raise RuntimeError("E2B is disabled. Set E2B_API_KEY to enable sandboxes.")
        
        self.logger.info(f"Running code in sandbox {sandbox_id}")
        result = await self.manager.execute_code(sandbox_id, code, timeout=timeout)
        return result
