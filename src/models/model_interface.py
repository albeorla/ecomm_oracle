from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class Model(ABC):
    """Abstract base class for all prediction models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            data: DataFrame containing training data
            
        Returns:
            Dictionary containing model performance metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            Series containing predictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model
        """
        pass 