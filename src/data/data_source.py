from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class DataSource(ABC):
    """Abstract interface for data sources."""

    @abstractmethod
    def get_training_data(self) -> pd.DataFrame:
        """
        Get historical training data.
        
        Returns:
            DataFrame with standardized schema for training
        """
        pass
    
    @abstractmethod
    def get_validation_data(self) -> pd.DataFrame:
        """
        Get validation data.
        
        Returns:
            DataFrame with standardized schema for validation
        """
        pass
    
    @abstractmethod
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """
        Validate data schema matches requirements.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if schema is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about features including types and constraints.
        
        Returns:
            Dictionary mapping feature names to their specifications
        """
        pass
    
    def validate_features(self, data: pd.DataFrame) -> bool:
        """
        Validate feature values match expected constraints.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if all features are valid, False otherwise
        """
        feature_info = self.get_feature_info()
        
        for feature, specs in feature_info.items():
            if feature not in data.columns:
                return False
            
            if 'type' in specs and not data[feature].dtype == specs['type']:
                return False
                
            if 'min' in specs and data[feature].min() < specs['min']:
                return False
                
            if 'max' in specs and data[feature].max() > specs['max']:
                return False
        
        return True