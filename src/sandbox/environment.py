"""
Sandbox Environment for the Autonomous ML Agent.

This module provides sandbox environment management
including setup and configuration.
"""

from typing import List
import logging

from config.settings import settings
from sandbox.e2b_manager import E2BManager

logger = logging.getLogger(__name__)

class SandboxEnvironment:
    """Sandbox environment management component."""
    
    def __init__(self, manager: E2BManager | None = None):
        """Initialize the sandbox environment."""
        self.logger = logger
        self.manager = manager or E2BManager(settings)
    
    async def setup_environment(self, sandbox_id: str) -> bool:
        """
        Setup sandbox environment.
        Ensures sandbox exists and is reachable.
        """
        if not self.manager.enabled:
            raise RuntimeError("E2B is disabled. Set E2B_API_KEY to enable sandboxes.")
        
        status = await self.manager.get_sandbox_status(sandbox_id)
        self.logger.info(f"Sandbox {sandbox_id} status: {status.get('status')}")
        return True
    
    async def install_dependencies(self, sandbox_id: str, packages: List[str]) -> bool:
        """Install required packages in sandbox."""
        if not self.manager.enabled:
            raise RuntimeError("E2B is disabled. Set E2B_API_KEY to enable sandboxes.")
        
        all_ok = True
        for pkg in packages:
            ok = await self.manager.install_package(sandbox_id, pkg)
            all_ok = all_ok and ok
        return all_ok
