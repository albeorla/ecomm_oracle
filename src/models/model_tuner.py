import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any
from loguru import logger
from sklearn.model_selection import cross_val_score

from .base_model import BaseModel

class ModelTuner:
    """Hyperparameter tuner using Optuna."""
    
    def __init__(
        self,
        model: BaseModel,
        n_trials: int = 20,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.model = model
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for Optuna optimization."""
        # Get hyperparameter suggestions based on model type
        if hasattr(self.model, 'get_param_space'):
            params = self.model.get_param_space(trial)
        else:
            # Default RandomForest parameters if not specified
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        
        # Set the parameters
        self.model.set_params(**params)
        
        # Perform cross-validation
        try:
            scores = cross_val_score(
                self.model.model,
                X,
                y,
                cv=self.cv_folds,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            # Convert MAE to positive score (higher is better for optuna)
            mean_score = -np.mean(scores)
            
            # Log intermediate results
            logger.debug(f"Trial {trial.number}: MAE = {mean_score:.4f}")
            
            return mean_score
            
        except Exception as e:
            logger.warning(f"Trial failed with parameters {params}: {str(e)}")
            # Return a poor score to discourage these parameters
            return float('inf')
    
    def tune(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            Dictionary of best parameters
        """
        logger.info("Starting hyperparameter tuning")
        
        # Prepare data
        X = data.drop('monthly_profit', axis=1)
        y = data['monthly_profit']
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best MAE: {study.best_value:.4f}")
        
        # Log optimization history
        try:
            # Create optimization history plot
            optuna.visualization.plot_optimization_history(study)
            # You could save this plot if needed
        except Exception as e:
            logger.warning(f"Failed to create optimization plot: {e}")
        
        return best_params