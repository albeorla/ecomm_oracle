import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model_interface import Model
from .model_tuner import ModelTuner

class RandomForestModel(Model):
    """Random Forest implementation for profitability prediction."""
    
    def __init__(
        self,
        model_dir: str = "models",
        feature_columns: List[str] = None,
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            model_dir: Directory to save/load models from
            feature_columns: List of feature column names to use
            random_state: Random seed for reproducibility
            **model_params: Parameters to pass to RandomForestRegressor
        """
        self.model = RandomForestRegressor(
            random_state=random_state,
            **model_params
        )
        self.scaler = StandardScaler()
        self.feature_columns = feature_columns or [
            'price', 'weight', 'review_rating', 'review_count',
            'competitors', 'estimated_monthly_sales', 'fba_fees', 'cogs'
        ]
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_model_path(self) -> Path:
        """Get the default path for saving/loading the model."""
        return self.model_dir / "random_forest_model.joblib"
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training or prediction."""
        # One-hot encode categorical variables if present
        categorical_columns = ['category', 'subcategory']
        X = df.copy()
        
        for col in categorical_columns:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], prefix=col)
        
        # Get all feature columns including one-hot encoded ones
        feature_cols = (
            self.feature_columns +
            [col for col in X.columns if any(col.startswith(f"{c}_") for c in categorical_columns)]
        )
        
        # Select only the feature columns that exist
        available_cols = [col for col in feature_cols if col in X.columns]
        return X[available_cols]
    
    def tune(
        self,
        data: pd.DataFrame,
        n_trials: int = 100,
        n_folds: int = 5,
        study_name: str = "random_forest_optimization"
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using cross-validation.
        
        Args:
            data: Training data
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            study_name: Name for the optimization study
            
        Returns:
            Dictionary containing optimization results
        """
        print("\nTuning model hyperparameters...")
        X = self._preprocess_data(data)
        y = data['monthly_profit']
        
        # Create and run tuner
        tuner = ModelTuner(
            n_trials=n_trials,
            n_folds=n_folds,
            random_state=42,
            study_name=study_name,
            storage_dir=self.model_dir
        )
        
        results = tuner.tune_model(X, y, RandomForestRegressor)
        
        # Update model with best parameters
        self.model = RandomForestRegressor(
            **results['best_params'],
            random_state=42
        )
        
        return results
    
    def train(
        self,
        data: pd.DataFrame,
        auto_save: bool = True,
        tune_first: bool = False,
        **tune_params
    ) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            data: DataFrame containing training data
            auto_save: Whether to automatically save the model after training
            tune_first: Whether to tune hyperparameters before training
            **tune_params: Parameters to pass to tune() if tune_first is True
            
        Returns:
            Dictionary containing model performance metrics
        """
        # Tune hyperparameters if requested
        if tune_first:
            self.tune(data, **tune_params)
        
        X = self._preprocess_data(data)
        y = data['monthly_profit']
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on validation set
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        # Auto-save if enabled
        if auto_save:
            self.save()
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            Series containing predictions
        """
        X = self._preprocess_data(data)
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict(X_scaled))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        return dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model. If None, uses default path.
        """
        save_path = Path(path) if path else self._get_default_model_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model. If None, uses default path.
        """
        load_path = Path(path) if path else self._get_default_model_path()
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model found at {load_path}")
            
        model_data = joblib.load(load_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {load_path}") 