#!/usr/bin/env python3
"""
Test script for LLM Integration (Phase 4).
Skips real calls if GEMINI_API_KEY is not set; verifies fallbacks.
"""

import os
import sys
from pathlib import Path
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_settings
from llm.code_generator import CodeGenerator
from llm.prompt_templates import PromptTemplates

async def main():
    settings = get_settings()
    gen = CodeGenerator()

    if not settings.GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not set. Verifying fallback behavior.")
        code = await gen.generate_preprocessing_code({"columns": ["a", "b"]})
        assert "Fallback preprocessing" in code
        print("✅ Fallback preprocessing code generated")
        return

    # If API key is present, try a light call
    code = await gen.generate_preprocessing_code({"columns": ["a", "b"]})
    print("✅ LLM generated preprocessing code (truncated):", code[:120])

if __name__ == "__main__":
    asyncio.run(main())
