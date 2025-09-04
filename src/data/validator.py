"""
Data validation and analysis utilities for the Autonomous ML Agent.

This module provides comprehensive dataset analysis, validation,
and quality assessment capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Provides data validation and analysis functionality."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            df: Input DataFrame to analyze
            target_column: Optional target column for supervised learning analysis
            
        Returns:
            Dictionary containing comprehensive dataset analysis
        """
        self.logger.info(f"Starting dataset analysis for DataFrame with shape: {df.shape}")
        
        analysis = {
            'basic_info': self._get_basic_info(df),
            'data_types': self._analyze_data_types(df),
            'missing_values': self._analyze_missing_values(df),
            'numerical_analysis': self._analyze_numerical_columns(df),
            'categorical_analysis': self._analyze_categorical_columns(df),
            'datetime_analysis': self._analyze_datetime_columns(df),
            'duplicates': self._analyze_duplicates(df),
            'outliers': self._analyze_outliers(df),
            'correlations': self._analyze_correlations(df),
            'data_quality_score': self._calculate_data_quality_score(df)
        }
        
        # Add target-specific analysis if target column is provided
        if target_column and target_column in df.columns:
            analysis['target_analysis'] = self._analyze_target_column(df, target_column)
        
        self.logger.info("Dataset analysis completed successfully")
        return analysis
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            'shape': df.shape,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            'column_names': list(df.columns)
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types of all columns."""
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_details = {}
        
        for col in df.columns:
            dtype_details[col] = {
                'dtype': str(df[col].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': pd.api.types.is_categorical_dtype(df[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(df[col]),
                'is_object': pd.api.types.is_object_dtype(df[col])
            }
        
        return {
            'dtype_counts': dtype_counts,
            'dtype_details': dtype_details
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_analysis = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': round((missing_counts.sum() / (len(df) * len(df.columns))) * 100, 2),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'complete_columns': list(missing_counts[missing_counts == 0].index)
        }
        
        return missing_analysis
    
    def _analyze_numerical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'count': 0, 'columns': [], 'statistics': {}}
        
        statistics = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                statistics[col] = {
                    'count': len(col_data),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
        
        return {
            'count': len(numerical_cols),
            'columns': list(numerical_cols),
            'statistics': statistics
        }
    
    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {'count': 0, 'columns': [], 'statistics': {}}
        
        statistics = {}
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                statistics[col] = {
                    'count': len(col_data),
                    'unique_values': col_data.nunique(),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_common_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict()
                }
        
        return {
            'count': len(categorical_cols),
            'columns': list(categorical_cols),
            'statistics': statistics
        }
    
    def _analyze_datetime_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze datetime columns."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return {'count': 0, 'columns': [], 'statistics': {}}
        
        statistics = {}
        for col in datetime_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                statistics[col] = {
                    'count': len(col_data),
                    'min_date': col_data.min().isoformat(),
                    'max_date': col_data.max().isoformat(),
                    'date_range_days': (col_data.max() - col_data.min()).days,
                    'unique_dates': col_data.nunique()
                }
        
        return {
            'count': len(datetime_cols),
            'columns': list(datetime_cols),
            'statistics': statistics
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        
        return {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': round(duplicate_percentage, 2),
            'unique_rows': len(df) - duplicate_rows
        }
    
    def _analyze_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """Analyze outliers in numerical columns using IQR method."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'count': 0, 'columns': {}, 'total_outliers': 0}
        
        outlier_analysis = {}
        total_outliers = 0
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count = len(outliers)
                total_outliers += outlier_count
                
                outlier_analysis[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': round((outlier_count / len(col_data)) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return {
            'count': len(numerical_cols),
            'columns': outlier_analysis,
            'total_outliers': total_outliers
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {'count': len(numerical_cols), 'correlation_matrix': None}
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find high correlations (|r| > 0.8)
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_correlations.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        return {
            'count': len(numerical_cols),
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality score."""
        total_score = 0
        max_score = 100
        details = {}
        
        # Missing values score (25 points)
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        missing_score = max(0, 25 - (missing_percentage * 0.5))
        total_score += missing_score
        details['missing_values_score'] = round(missing_score, 2)
        
        # Duplicates score (20 points)
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        duplicate_score = max(0, 20 - (duplicate_percentage * 0.4))
        total_score += duplicate_score
        details['duplicates_score'] = round(duplicate_score, 2)
        
        # Data types score (20 points)
        object_cols = df.select_dtypes(include=['object']).columns
        dtype_score = max(0, 20 - (len(object_cols) * 2))
        total_score += dtype_score
        details['data_types_score'] = round(dtype_score, 2)
        
        # Outliers score (15 points)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            outlier_percentage = 0
            for col in numerical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                    outlier_percentage += len(outliers) / len(col_data)
            
            outlier_percentage = outlier_percentage / len(numerical_cols) * 100
            outlier_score = max(0, 15 - (outlier_percentage * 0.3))
            total_score += outlier_score
            details['outliers_score'] = round(outlier_score, 2)
        else:
            details['outliers_score'] = 15
        
        # Consistency score (20 points)
        consistency_score = 20  # Placeholder - could be enhanced with domain-specific rules
        total_score += consistency_score
        details['consistency_score'] = consistency_score
        
        return {
            'overall_score': round(total_score, 2),
            'max_score': max_score,
            'grade': self._get_grade(total_score),
            'details': details
        }
    
    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
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
    
    def _analyze_target_column(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze the target column for supervised learning."""
        if target_column not in df.columns:
            return {'error': f'Target column {target_column} not found in dataset'}
        
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
                    'median': float(target_data.median())
                },
                'distribution': 'normal' if abs(target_data.skew()) < 1 else 'skewed'
            }
        else:
            # Classification target
            value_counts = target_data.value_counts()
            return {
                'type': 'classification',
                'unique_classes': target_data.nunique(),
                'class_distribution': value_counts.to_dict(),
                'class_balance': 'balanced' if len(value_counts) <= 2 else 'imbalanced',
                'majority_class': value_counts.index[0],
                'majority_percentage': round((value_counts.iloc[0] / len(target_data)) * 100, 2)
            }
    
    def validate_features(self, df: pd.DataFrame, required_features: List[str]) -> Dict[str, Any]:
        """
        Validate that required features exist and meet quality criteria.
        
        Args:
            df: Input DataFrame
            required_features: List of required feature names
            
        Returns:
            Validation results
        """
        validation_results = {
            'all_features_present': True,
            'missing_features': [],
            'feature_quality': {},
            'overall_valid': True
        }
        
        # Check if all required features are present
        for feature in required_features:
            if feature not in df.columns:
                validation_results['missing_features'].append(feature)
                validation_results['all_features_present'] = False
                validation_results['overall_valid'] = False
        
        # Analyze quality of present features
        for feature in required_features:
            if feature in df.columns:
                feature_data = df[feature].dropna()
                quality_score = 100
                
                # Missing values penalty
                missing_percentage = (df[feature].isnull().sum() / len(df)) * 100
                quality_score -= missing_percentage * 0.5
                
                # Duplicates penalty (for categorical)
                if pd.api.types.is_object_dtype(df[feature]):
                    duplicate_percentage = (df[feature].duplicated().sum() / len(df)) * 100
                    quality_score -= duplicate_percentage * 0.3
                
                # Outliers penalty (for numerical)
                if pd.api.types.is_numeric_dtype(df[feature]):
                    if len(feature_data) > 0:
                        Q1 = feature_data.quantile(0.25)
                        Q3 = feature_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = feature_data[(feature_data < Q1 - 1.5 * IQR) | (feature_data > Q3 + 1.5 * IQR)]
                        outlier_percentage = (len(outliers) / len(feature_data)) * 100
                        quality_score -= outlier_percentage * 0.2
                
                quality_score = max(0, quality_score)
                validation_results['feature_quality'][feature] = round(quality_score, 2)
                
                if quality_score < 70:
                    validation_results['overall_valid'] = False
        
        return validation_results
