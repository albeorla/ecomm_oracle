from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.target_column = 'monthly_profit'
    
    def _prepare_data(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        if self.feature_columns is None:
            # Use all columns except target for features
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        X = data[self.feature_columns].values
        if self.target_column in data.columns:
            y = data[self.target_column].values
            return X, y
        return X
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model and return metrics.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            Dictionary of training metrics
        """
        X, y = self._prepare_data(data)
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self._prepare_data(data)[0]
        return self.model.predict(X)
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, y = self._prepare_data(data)
        y_pred = self.model.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and feature columns
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, path)
    
    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file found at {path}")
        
        # Load model and feature columns
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.feature_columns = saved_data['feature_columns']
        self.target_column = saved_data['target_column']
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        if self.model is None:
            raise ValueError("Model not initialized. Initialize the model first.")
        self.model.set_params(**params)
    
    @abstractmethod
    def get_param_space(self, trial) -> Dict[str, Any]:
        """Define the hyperparameter search space for the model."""
        pass