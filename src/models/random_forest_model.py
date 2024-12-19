import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import optuna

from .model_interface import Model
from .model_tuner import ModelTuner
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation for profitability prediction."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the hyperparameter search space for Random Forest."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
    
    def train(self, df, tune_first: bool = False):
        """Train the Random Forest model.
        
        Args:
            df (pd.DataFrame): Training data
            tune_first (bool): Whether to tune hyperparameters before training
            
        Returns:
            dict: Training metrics
        """
        # Store categories from training data
        for cat_col in self.categorical_columns:
            if cat_col in df.columns:
                if not hasattr(self, 'categories_'):
                    self.categories_ = {}
                self.categories_[cat_col] = sorted(df[cat_col].unique().tolist())
        
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        if tune_first:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
        
        return self.evaluate(X_test, y_test)
    
    def calculate_business_metrics(self, df, predictions):
        """Calculate business-specific metrics.
        
        Args:
            df (pd.DataFrame): Input data with actual values
            predictions (np.array): Model predictions
            
        Returns:
            dict: Business metrics
        """
        # Convert predictions to numpy array if needed
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Calculate monthly values
        monthly_investment = df['cogs'] * df['estimated_monthly_sales']
        monthly_revenue = df['price'] * df['estimated_monthly_sales']
        
        # Normalize data for better correlation calculation
        norm_competitors = (df['competitors'] - df['competitors'].mean()) / df['competitors'].std()
        norm_reviews = (df['review_rating'] - df['review_rating'].mean()) / df['review_rating'].std()
        norm_predictions = (predictions - predictions.mean()) / predictions.std()
        
        # Calculate correlations with error handling
        try:
            competitor_corr = np.corrcoef(norm_competitors, norm_predictions)[0, 1]
            review_corr = np.corrcoef(norm_reviews, norm_predictions)[0, 1]
        except:
            competitor_corr = review_corr = 0
        
        # Calculate metrics
        metrics = {
            'average_roi': float((np.mean(predictions) / np.mean(monthly_investment)) * 100),
            'average_profit_margin': float((np.mean(predictions) / np.mean(monthly_revenue)) * 100),
            'break_even_units': max(1, float(np.ceil(df['cogs'].mean() / (df['price'].mean() - df['fba_fees'].mean())))),
            'average_monthly_sales': float(df['estimated_monthly_sales'].mean()),
            'payback_period_months': float(np.mean(monthly_investment) / np.mean(predictions)),
            'competitor_impact': float(competitor_corr if not np.isnan(competitor_corr) else 0),
            'review_impact': float(review_corr if not np.isnan(review_corr) else 0)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """Get the importance of each feature in the model.
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get feature names including encoded categorical features
        X = self.preprocess_data(pd.DataFrame({
            col: [0] for col in self.feature_columns
        }))
        feature_names = X.columns.tolist()
        
        # Get feature importance scores
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _get_default_model_path(self) -> Path:
        """Get the default path for saving/loading the model."""
        return self.model_dir / "random_forest_model.joblib"
    
    def tune(
        self,
        data: pd.DataFrame,
        n_trials: int = 100,
        n_folds: int = 5,
        study_name: str = "random_forest_optimization"
    ) -> Dict[str, Any]:
        """Tune model hyperparameters using cross-validation.
        
        Args:
            data: Training data
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            study_name: Name for the optimization study
            
        Returns:
            Dictionary containing optimization results
        """
        print("\nTuning model hyperparameters...")
        
        # Store categories before preprocessing
        for cat_col in self.categorical_columns:
            if cat_col in data.columns:
                if not hasattr(self, 'categories_'):
                    self.categories_ = {}
                self.categories_[cat_col] = sorted(data[cat_col].unique().tolist())
        
        # Prepare data for tuning
        X = self.preprocess_data(data)
        y = data['monthly_profit'].astype(np.float64)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=n_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = RandomForestRegressor(
            **grid_search.best_params_,
            random_state=42
        )
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_  # Convert back to positive RMSE
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            df (pd.DataFrame): Data to make predictions on
            
        Returns:
            np.ndarray: Predicted values
        """
        # Use the base class's predict method
        return super().predict(df)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to disk.
        
        Args:
            path: Path where to save the model. If None, uses default path.
        """
        save_path = Path(path) if path else self._get_default_model_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'categories_': getattr(self, 'categories_', {})
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from disk.
        
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
        self.categorical_columns = model_data['categorical_columns']
        if 'categories_' in model_data:
            self.categories_ = model_data['categories_']
        print(f"Model loaded from {load_path}") 