import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from loguru import logger

from .model_interface import Model
from .model_tuner import ModelTuner

class GradientBoostModel(Model):
    """XGBoost implementation for profitability prediction."""
    
    def __init__(
        self,
        model_dir: str = "models",
        feature_columns: List[str] = None,
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            model_dir: Directory to save/load models from
            feature_columns: List of feature column names to use
            random_state: Random seed for reproducibility
            **model_params: Parameters to pass to XGBRegressor
        """
        try:
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state,
                'n_jobs': -1  # Use all CPU cores
            }
            default_params.update(model_params)
            
            self.model = XGBRegressor(**default_params)
            self.scaler = StandardScaler()
            self.feature_columns = feature_columns or [
                'price', 'weight', 'review_rating', 'review_count',
                'competitors', 'estimated_monthly_sales', 'fba_fees', 'cogs'
            ]
            self.model_dir = Path(model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize XGBoost model: {str(e)}", exc_info=True)
            raise
    
    def calculate_business_metrics(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate important business metrics for FBA products.
        
        Args:
            data: DataFrame containing product features
            predictions: Array of predicted monthly profits
            
        Returns:
            Dictionary containing calculated business metrics
        """
        try:
            metrics = {}
            
            # Convert predictions to pandas Series with matching index
            predictions_series = pd.Series(predictions, index=data.index)
            
            # Monthly revenue and costs
            monthly_sales = data['estimated_monthly_sales']
            unit_price = data['price']
            monthly_revenue = unit_price * monthly_sales
            
            # Monthly costs
            unit_costs = data['cogs'] + data['fba_fees']
            monthly_costs = unit_costs * monthly_sales
            
            # Monthly profit (using actual monthly sales data)
            monthly_profit = monthly_revenue - monthly_costs
            metrics['average_monthly_profit'] = float(monthly_profit.mean())
            
            # ROI calculation (monthly)
            monthly_investment = monthly_costs  # Initial investment is monthly inventory cost
            valid_investments = monthly_investment > 0
            if valid_investments.any():
                monthly_roi = (monthly_profit[valid_investments] / monthly_investment[valid_investments] * 100).mean()
                metrics['average_roi'] = float(monthly_roi)
            else:
                metrics['average_roi'] = 0.0
            
            # Profit Margins
            valid_revenue = monthly_revenue > 0
            if valid_revenue.any():
                profit_margin = (monthly_profit[valid_revenue] / monthly_revenue[valid_revenue] * 100).mean()
                metrics['average_profit_margin'] = float(profit_margin)
            else:
                metrics['average_profit_margin'] = 0.0
            
            # Break-even Analysis (units)
            margin_per_unit = unit_price - unit_costs
            valid_margins = margin_per_unit > 0.01  # Minimum 1 cent margin
            if valid_margins.any():
                break_even_units = (unit_costs[valid_margins] / margin_per_unit[valid_margins]).mean()
                metrics['break_even_units'] = float(break_even_units)
            else:
                metrics['break_even_units'] = float('inf')
            
            # Average Monthly Sales
            metrics['average_monthly_sales'] = float(monthly_sales.mean())
            
            # Payback Period (in months)
            MIN_MONTHLY_PROFIT = 0.01  # Minimum $0.01 monthly profit
            valid_profits = monthly_profit > MIN_MONTHLY_PROFIT
            if valid_profits.any():
                payback_period = (monthly_investment[valid_profits] / monthly_profit[valid_profits]).mean()
                payback_period = min(float(payback_period), 120)  # Cap at 10 years
            else:
                payback_period = float('inf')
            metrics['payback_period_months'] = payback_period
            
            # Net Profit after All Fees (monthly)
            metrics['average_net_profit'] = float(monthly_profit.mean())
            
            # Competition Analysis
            try:
                competitor_data = data['competitors'].astype(float)
                profit_data = monthly_profit.astype(float)
                valid_data = ~(pd.isna(competitor_data) | pd.isna(profit_data))
                if valid_data.any() and competitor_data[valid_data].std() > 0 and profit_data[valid_data].std() > 0:
                    competitor_correlation = competitor_data[valid_data].corr(profit_data[valid_data])
                    metrics['competitor_impact'] = float(competitor_correlation) if not pd.isna(competitor_correlation) else 0.0
                else:
                    metrics['competitor_impact'] = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate competitor impact: {str(e)}")
                metrics['competitor_impact'] = 0.0
            
            # Review Impact
            try:
                review_data = data['review_rating'].astype(float)
                profit_data = monthly_profit.astype(float)
                valid_data = ~(pd.isna(review_data) | pd.isna(profit_data))
                if valid_data.any() and review_data[valid_data].std() > 0 and profit_data[valid_data].std() > 0:
                    review_correlation = review_data[valid_data].corr(profit_data[valid_data])
                    metrics['review_impact'] = float(review_correlation) if not pd.isna(review_correlation) else 0.0
                else:
                    metrics['review_impact'] = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate review impact: {str(e)}")
                metrics['review_impact'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate business metrics: {str(e)}", exc_info=True)
            raise
    
    def _get_default_model_path(self) -> Path:
        """Get the default path for saving/loading the model."""
        return self.model_dir / "gradient_boost_model.joblib"
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training or prediction."""
        try:
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
            
            if not available_cols:
                raise ValueError("No valid feature columns found after preprocessing")
            
            return X[available_cols]
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
            raise
    
    def tune(
        self,
        data: pd.DataFrame,
        n_trials: int = 100,
        n_folds: int = 5,
        study_name: str = "gradient_boost_optimization"
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
        try:
            logger.info("\nTuning XGBoost hyperparameters...")
            X = self._preprocess_data(data)
            y = data['monthly_profit']
            
            # Create and run tuner with XGBoost-specific search space
            tuner = ModelTuner(
                n_trials=n_trials,
                n_folds=n_folds,
                random_state=42,
                study_name=study_name,
                storage_dir=self.model_dir
            )
            
            def xgb_search_space(trial):
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5)
                }
            
            tuner._get_search_space = xgb_search_space
            results = tuner.tune_model(X, y, XGBRegressor)
            
            # Update model with best parameters
            self.model = XGBRegressor(
                **results['best_params'],
                random_state=42,
                n_jobs=-1
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}", exc_info=True)
            raise
    
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
        try:
            # Validate input data
            required_columns = ['monthly_profit'] + self.feature_columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for NaN values
            nan_columns = data[required_columns].columns[data[required_columns].isna().any()].tolist()
            if nan_columns:
                raise ValueError(f"Found NaN values in columns: {nan_columns}")
            
            # Tune hyperparameters if requested
            if tune_first:
                self.tune(data, **tune_params)
            
            X = self._preprocess_data(data)
            y = data['monthly_profit']
            
            # Validate target values
            if y.isna().any():
                raise ValueError("Found NaN values in target variable 'monthly_profit'")
            if len(y.unique()) < 2:
                raise ValueError("Target variable 'monthly_profit' has insufficient variation")
            
            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Validate scaled data
            if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
                raise ValueError("Scaling produced NaN values")
            
            # Train the model with early stopping and progress tracking
            logger.info("Starting model training...")
            eval_set = [(X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
            logger.info("Model training completed")
            
            # Make predictions on validation set
            y_pred = self.model.predict(X_val_scaled)
            
            # Validate predictions
            if np.isnan(y_pred).any():
                raise ValueError("Model produced NaN predictions during validation")
            
            # Calculate standard metrics
            metrics = {
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred)
            }
            
            # Log key metrics
            logger.info(f"Training metrics - MAE: ${metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
            
            # Calculate prediction ranges for validation
            pred_range = {
                'min_pred': float(y_pred.min()),
                'max_pred': float(y_pred.max()),
                'mean_pred': float(y_pred.mean()),
                'median_pred': float(np.median(y_pred))
            }
            logger.info(f"Prediction range: ${pred_range['min_pred']:.2f} to ${pred_range['max_pred']:.2f}")
            
            # Add business metrics
            try:
                business_metrics = self.calculate_business_metrics(X_val, y_pred)
                metrics.update(business_metrics)
            except Exception as e:
                logger.warning(f"Business metrics calculation failed: {str(e)}")
                logger.warning("Continuing with standard metrics only")
            
            # Add prediction range metrics
            metrics.update(pred_range)
            
            # Auto-save if enabled
            if auto_save:
                try:
                    self.save()
                except Exception as e:
                    logger.warning(f"Model auto-save failed: {str(e)}")
                    logger.warning("Continuing without saving")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            Series containing predictions
        """
        try:
            # Validate input data
            required_columns = self.feature_columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for NaN values in input
            nan_columns = data[required_columns].columns[data[required_columns].isna().any()].tolist()
            if nan_columns:
                raise ValueError(f"Found NaN values in columns: {nan_columns}")
            
            # Preprocess and scale data
            X = self._preprocess_data(data)
            X_scaled = self.scaler.transform(X)
            
            # Validate scaled data
            if np.isnan(X_scaled).any():
                raise ValueError("Scaling produced NaN values")
            
            # Make predictions
            logger.debug("Making predictions...")
            predictions = self.model.predict(X_scaled)
            
            # Validate predictions
            if np.isnan(predictions).any():
                raise ValueError("Model produced NaN predictions")
            
            # Log prediction statistics
            logger.debug(f"Prediction range: ${predictions.min():.2f} to ${predictions.max():.2f}")
            logger.debug(f"Mean prediction: ${predictions.mean():.2f}")
            
            return pd.Series(predictions, index=data.index)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        try:
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            return dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}", exc_info=True)
            raise
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model. If None, uses default path.
        """
        try:
            save_path = Path(path) if path else self._get_default_model_path()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model. If None, uses default path.
        """
        try:
            load_path = Path(path) if path else self._get_default_model_path()
            if not load_path.exists():
                raise FileNotFoundError(f"No saved model found at {load_path}")
                
            model_data = joblib.load(load_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise 