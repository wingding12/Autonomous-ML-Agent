#!/usr/bin/env python3
"""
Test script for E2B Sandbox Integration (Phase 3).
- Creates a sandbox
- Runs simple code
- Installs a package
- Cleans up
Skips gracefully if E2B_API_KEY is not set.
"""

import os
import sys
from pathlib import Path
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_settings
from sandbox.e2b_manager import E2BManager
from sandbox.code_runner import CodeRunner
from sandbox.environment import SandboxEnvironment

async def main():
    settings = get_settings()
    manager = E2BManager(settings)

    if not manager.enabled:
        print("⚠️  E2B_API_KEY not set. Skipping sandbox tests.")
        return

    sandbox_id = "test-sbx"

    try:
        # Create sandbox
        await manager.create_sandbox(sandbox_id)
        print("✅ Sandbox created")

        # Status
        status = await manager.get_sandbox_status(sandbox_id)
        print(f"✅ Sandbox status: {status['status']}")

        # Run code
        runner = CodeRunner(manager)
        result = await runner.run_code("print('hello from sandbox')", sandbox_id)
        assert result["success"], result
        print("✅ Code executed: ", result["stdout"].strip())

        # Install package
        env = SandboxEnvironment(manager)
        ok = await env.install_dependencies(sandbox_id, ["numpy"])
        print("✅ Package install status:", ok)

    finally:
        # Cleanup
        await manager.terminate_sandbox(sandbox_id)
        print("✅ Sandbox terminated")

if __name__ == "__main__":
    asyncio.run(main())
