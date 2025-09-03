"""
File Utilities for the Autonomous ML Agent.

This module provides file handling utilities including
file operations and path management.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class FileUtils:
    """File utility component."""
    
    def __init__(self):
        """Initialize the file utilities."""
        self.logger = logger
    
    async def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Save data as JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
            return False
    
    async def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            return None
