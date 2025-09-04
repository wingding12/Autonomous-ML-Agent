"""
Data loading utilities for the Autonomous ML Agent.

This module handles loading and parsing of various data formats,
including CSV, Excel, Parquet, and JSON files.
"""

import os
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and parsing of various data formats."""
    
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.parquet': 'parquet',
        '.json': 'json',
        '.pkl': 'pickle',
        '.h5': 'hdf5'
    }
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load a dataset from various file formats.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded dataset as pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            Exception: For other loading errors
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension and validate format
        file_extension = file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        try:
            self.logger.info(f"Loading dataset from: {file_path}")
            
            # Load based on file format
            if file_extension == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_extension == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_extension == '.pkl':
                df = pd.read_pickle(file_path, **kwargs)
            elif file_extension == '.h5':
                df = pd.read_hdf(file_path, **kwargs)
            else:
                raise ValueError(f"Unhandled file format: {file_extension}")
            
            self.logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset from {file_path}: {str(e)}")
            raise
    
    def validate_format(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate if a file format is supported.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, format_type)
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension in self.SUPPORTED_FORMATS:
            return True, self.SUPPORTED_FORMATS[file_extension]
        else:
            return False, "unsupported"
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a data file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get basic file info
        stat = file_path.stat()
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': stat.st_size,
            'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
            'file_extension': file_path.suffix.lower(),
            'last_modified': stat.st_mtime,
            'is_supported': file_path.suffix.lower() in self.SUPPORTED_FORMATS
        }
        
        # Try to get dataset info if supported
        if file_info['is_supported']:
            try:
                # Load just a sample to get basic info
                sample_df = self.load_dataset(file_path, nrows=1000)
                file_info.update({
                    'sample_rows': len(sample_df),
                    'sample_columns': len(sample_df.columns),
                    'column_names': list(sample_df.columns),
                    'dtypes': sample_df.dtypes.to_dict()
                })
            except Exception as e:
                self.logger.warning(f"Could not load sample from {file_path}: {str(e)}")
                file_info['load_error'] = str(e)
        
        return file_info
    
    def list_supported_formats(self) -> Dict[str, str]:
        """
        Get list of supported file formats.
        
        Returns:
            Dictionary mapping file extensions to format names
        """
        return self.SUPPORTED_FORMATS.copy()
    
    def load_sample(self, file_path: Union[str, Path], nrows: int = 1000, **kwargs) -> pd.DataFrame:
        """
        Load a sample of the dataset for quick inspection.
        
        Args:
            file_path: Path to the data file
            nrows: Number of rows to load
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Sample dataset as pandas DataFrame
        """
        # Add nrows parameter for supported formats
        if 'nrows' not in kwargs:
            kwargs['nrows'] = nrows
        
        return self.load_dataset(file_path, **kwargs)
