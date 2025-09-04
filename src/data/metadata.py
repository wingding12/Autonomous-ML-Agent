"""
Data metadata management for the Autonomous ML Agent.

This module handles extraction, storage, and retrieval of dataset metadata
for pipeline optimization and meta-learning.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataMetadata:
    """Handles data metadata management and storage."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the DataMetadata manager.
        
        Args:
            storage_dir: Directory to store metadata files (default: ./metadata)
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./metadata")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file paths
        self.dataset_metadata_file = self.storage_dir / "dataset_metadata.json"
        self.pipeline_metadata_file = self.storage_dir / "pipeline_metadata.json"
        self.experiment_metadata_file = self.storage_dir / "experiment_metadata.json"
        
        # Load existing metadata
        self.dataset_metadata = self._load_metadata(self.dataset_metadata_file)
        self.pipeline_metadata = self._load_metadata(self.pipeline_metadata_file)
        self.experiment_metadata = self._load_metadata(self.experiment_metadata_file)
    
    def extract_metadata(self, df: pd.DataFrame, dataset_name: str, 
                        target_column: Optional[str] = None,
                        additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a dataset.
        
        Args:
            df: Input DataFrame
            dataset_name: Name/identifier for the dataset
            target_column: Optional target column name
            additional_info: Additional metadata to include
            
        Returns:
            Dictionary containing extracted metadata
        """
        self.logger.info(f"Extracting metadata for dataset: {dataset_name}")
        
        # Generate dataset hash for identification
        dataset_hash = self._generate_dataset_hash(df)
        
        # Basic dataset information
        metadata = {
            'dataset_name': dataset_name,
            'dataset_hash': dataset_hash,
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'shape': df.shape,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': round((df.duplicated().sum() / len(df)) * 100, 2)
        }
        
        # Column type analysis
        metadata.update(self._analyze_column_types(df))
        
        # Statistical analysis
        metadata.update(self._analyze_statistics(df))
        
        # Target column analysis
        if target_column and target_column in df.columns:
            metadata['target_analysis'] = self._analyze_target_column(df, target_column)
        
        # Additional information
        if additional_info:
            metadata['additional_info'] = additional_info
        
        # Data quality metrics
        metadata['data_quality'] = self._calculate_data_quality_metrics(df)
        
        # Feature engineering potential
        metadata['feature_engineering_potential'] = self._assess_feature_engineering_potential(df)
        
        self.logger.info(f"Metadata extraction completed for {dataset_name}")
        return metadata
    
    def _generate_dataset_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset based on shape and column names."""
        # Create a string representation of dataset structure
        structure_str = f"{df.shape}_{'_'.join(sorted(df.columns))}_{len(df)}"
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    def _analyze_column_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and their characteristics."""
        column_analysis = {
            'numerical_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'boolean_columns': []
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                column_analysis['numerical_columns'].append({
                    'name': col,
                    'unique_values': col_data.nunique(),
                    'has_negative': bool((col_data < 0).any()) if len(col_data) > 0 else False,
                    'has_zero': bool((col_data == 0).any()) if len(col_data) > 0 else False,
                    'is_integer': pd.api.types.is_integer_dtype(df[col])
                })
            elif pd.api.types.is_categorical_dtype(df[col]):
                column_analysis['categorical_columns'].append({
                    'name': col,
                    'unique_values': col_data.nunique(),
                    'cardinality': 'low' if col_data.nunique() <= 10 else 'medium' if col_data.nunique() <= 100 else 'high'
                })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_analysis['datetime_columns'].append({
                    'name': col,
                    'unique_values': col_data.nunique(),
                    'date_range_days': (col_data.max() - col_data.min()).days if len(col_data) > 0 else 0
                })
            elif pd.api.types.is_object_dtype(df[col]):
                # Check if it's text or categorical
                avg_length = col_data.astype(str).str.len().mean()
                if pd.isna(avg_length) or avg_length > 50:
                    column_analysis['text_columns'].append({
                        'name': col,
                        'unique_values': col_data.nunique(),
                        'avg_length': round(avg_length, 2) if not pd.isna(avg_length) else 0
                    })
                else:
                    column_analysis['categorical_columns'].append({
                        'name': col,
                        'unique_values': col_data.nunique(),
                        'cardinality': 'low' if col_data.nunique() <= 10 else 'medium' if col_data.nunique() <= 100 else 'high'
                    })
            elif pd.api.types.is_bool_dtype(df[col]):
                column_analysis['boolean_columns'].append({
                    'name': col,
                    'true_percentage': round((col_data.sum() / len(col_data)) * 100, 2) if len(col_data) > 0 else 0
                })
        
        return column_analysis
    
    def _analyze_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical properties of the dataset."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'statistical_summary': {}}
        
        statistical_summary = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                statistical_summary[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis()),
                    'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                }
        
        return {'statistical_summary': statistical_summary}
    
    def _analyze_target_column(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze the target column for supervised learning."""
        target_data = df[target_column].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            # Regression target
            return {
                'type': 'regression',
                'statistics': {
                    'mean': float(target_data.mean()),
                    'std': float(target_data.std()),
                    'min': float(target_data.min()),
                    'max': float(target_data.max()),
                    'median': float(target_data.median()),
                    'skewness': float(target_data.skew()),
                    'kurtosis': float(target_data.kurtosis())
                },
                'distribution': 'normal' if abs(target_data.skew()) < 1 else 'skewed',
                'outliers': self._count_outliers(target_data)
            }
        else:
            # Classification target
            value_counts = target_data.value_counts()
            class_balance = self._assess_class_balance(value_counts)
            
            return {
                'type': 'classification',
                'unique_classes': target_data.nunique(),
                'class_distribution': value_counts.to_dict(),
                'class_balance': class_balance,
                'majority_class': value_counts.index[0] if len(value_counts) > 0 else None,
                'majority_percentage': round((value_counts.iloc[0] / len(target_data)) * 100, 2) if len(value_counts) > 0 else 0,
                'minority_class': value_counts.index[-1] if len(value_counts) > 0 else None,
                'minority_percentage': round((value_counts.iloc[-1] / len(target_data)) * 100, 2) if len(value_counts) > 0 else 0
            }
    
    def _count_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """Count outliers using IQR method."""
        if len(data) == 0:
            return {'count': 0, 'percentage': 0}
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(data)) * 100, 2)
        }
    
    def _assess_class_balance(self, value_counts: pd.Series) -> str:
        """Assess class balance in classification target."""
        if len(value_counts) <= 2:
            # Binary classification
            min_count = value_counts.min()
            max_count = value_counts.max()
            ratio = min_count / max_count
            if ratio >= 0.4:
                return 'balanced'
            elif ratio >= 0.2:
                return 'moderately_imbalanced'
            else:
                return 'highly_imbalanced'
        else:
            # Multi-class classification
            min_count = value_counts.min()
            max_count = value_counts.max()
            ratio = min_count / max_count
            if ratio >= 0.3:
                return 'balanced'
            elif ratio >= 0.1:
                return 'moderately_imbalanced'
            else:
                return 'highly_imbalanced'
    
    def _calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Calculate quality scores
        completeness_score = (total_cells - missing_cells) / total_cells * 100
        uniqueness_score = (len(df) - duplicate_rows) / len(df) * 100
        
        # Data type consistency
        object_cols = df.select_dtypes(include=['object']).columns
        dtype_consistency = (len(df.columns) - len(object_cols)) / len(df.columns) * 100
        
        # Overall quality score
        overall_quality = (completeness_score + uniqueness_score + dtype_consistency) / 3
        
        return {
            'completeness_score': round(completeness_score, 2),
            'uniqueness_score': round(uniqueness_score, 2),
            'dtype_consistency_score': round(dtype_consistency, 2),
            'overall_quality_score': round(overall_quality, 2),
            'grade': self._get_quality_grade(overall_quality)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _assess_feature_engineering_potential(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess potential for feature engineering."""
        potential = {
            'datetime_features': 0,
            'categorical_features': 0,
            'numerical_features': 0,
            'text_features': 0,
            'recommendations': []
        }
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                potential['datetime_features'] += 1
                potential['recommendations'].append(f"Extract date components from {col}")
            elif pd.api.types.is_object_dtype(df[col]):
                if df[col].nunique() <= 100:  # Likely categorical
                    potential['categorical_features'] += 1
                    potential['recommendations'].append(f"Encode {col} as categorical")
                else:  # Likely text
                    potential['text_features'] += 1
                    potential['recommendations'].append(f"Extract text features from {col}")
            elif pd.api.types.is_numeric_dtype(df[col]):
                potential['numerical_features'] += 1
                if df[col].nunique() <= 20:
                    potential['recommendations'].append(f"Consider {col} as categorical")
        
        return potential
    
    def store_metadata(self, metadata: Dict[str, Any], metadata_type: str = "dataset") -> bool:
        """
        Store metadata to the appropriate file.
        
        Args:
            metadata: Metadata dictionary to store
            metadata_type: Type of metadata ('dataset', 'pipeline', 'experiment')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metadata_type == "dataset":
                dataset_hash = metadata.get('dataset_hash')
                if dataset_hash:
                    self.dataset_metadata[dataset_hash] = metadata
                    self._save_metadata(self.dataset_metadata_file, self.dataset_metadata)
            elif metadata_type == "pipeline":
                pipeline_id = metadata.get('pipeline_id')
                if pipeline_id:
                    self.pipeline_metadata[pipeline_id] = metadata
                    self._save_metadata(self.pipeline_metadata_file, self.pipeline_metadata)
            elif metadata_type == "experiment":
                experiment_id = metadata.get('experiment_id')
                if experiment_id:
                    self.experiment_metadata[experiment_id] = metadata
                    self._save_metadata(self.experiment_metadata_file, self.experiment_metadata)
            else:
                self.logger.warning(f"Unknown metadata type: {metadata_type}")
                return False
            
            self.logger.info(f"Successfully stored {metadata_type} metadata")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing {metadata_type} metadata: {str(e)}")
            return False
    
    def get_metadata(self, identifier: str, metadata_type: str = "dataset") -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata by identifier.
        
        Args:
            identifier: Dataset hash, pipeline ID, or experiment ID
            metadata_type: Type of metadata to retrieve
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            if metadata_type == "dataset":
                return self.dataset_metadata.get(identifier)
            elif metadata_type == "pipeline":
                return self.pipeline_metadata.get(identifier)
            elif metadata_type == "experiment":
                return self.experiment_metadata.get(identifier)
            else:
                self.logger.warning(f"Unknown metadata type: {metadata_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving {metadata_type} metadata: {str(e)}")
            return None
    
    def find_similar_datasets(self, target_metadata: Dict[str, Any], 
                            similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find datasets similar to the target based on metadata.
        
        Args:
            target_metadata: Metadata of the target dataset
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar datasets with similarity scores
        """
        similar_datasets = []
        
        for dataset_hash, dataset_meta in self.dataset_metadata.items():
            similarity_score = self._calculate_similarity(target_metadata, dataset_meta)
            if similarity_score >= similarity_threshold:
                similar_datasets.append({
                    'dataset_hash': dataset_hash,
                    'dataset_name': dataset_meta.get('dataset_name', 'Unknown'),
                    'similarity_score': similarity_score,
                    'metadata': dataset_meta
                })
        
        # Sort by similarity score (descending)
        similar_datasets.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_datasets
    
    def _calculate_similarity(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> float:
        """Calculate similarity between two metadata dictionaries."""
        # Simple similarity calculation based on key features
        # This could be enhanced with more sophisticated algorithms
        
        score = 0.0
        total_weight = 0.0
        
        # Shape similarity (weight: 0.3)
        if 'shape' in metadata1 and 'shape' in metadata2:
            shape_diff = abs(metadata1['shape'][0] - metadata2['shape'][0]) / max(metadata1['shape'][0], 1)
            shape_score = max(0, 1 - shape_diff)
            score += shape_score * 0.3
            total_weight += 0.3
        
        # Column count similarity (weight: 0.2)
        if 'total_columns' in metadata1 and 'total_columns' in metadata2:
            col_diff = abs(metadata1['total_columns'] - metadata2['total_columns']) / max(metadata1['total_columns'], 1)
            col_score = max(0, 1 - col_diff)
            score += col_score * 0.2
            total_weight += 0.2
        
        # Data type similarity (weight: 0.3)
        if 'data_types' in metadata1 and 'data_types' in metadata2:
            type_matches = 0
            total_cols = 0
            for col in metadata1['data_types']:
                if col in metadata2['data_types']:
                    if metadata1['data_types'][col] == metadata2['data_types'][col]:
                        type_matches += 1
                    total_cols += 1
            
            if total_cols > 0:
                type_score = type_matches / total_cols
                score += type_score * 0.3
                total_weight += 0.3
        
        # Target type similarity (weight: 0.2)
        if 'target_analysis' in metadata1 and 'target_analysis' in metadata2:
            if metadata1['target_analysis'].get('type') == metadata2['target_analysis'].get('type'):
                score += 0.2
                total_weight += 0.2
        
        # Normalize score
        if total_weight > 0:
            return score / total_weight
        else:
            return 0.0
    
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"Could not load metadata from {file_path}: {str(e)}")
            return {}
    
    def _save_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metadata to {file_path}: {str(e)}")
            raise
