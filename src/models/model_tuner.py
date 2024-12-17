from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
from pathlib import Path
import json

class ModelTuner:
    """Tune model hyperparameters using cross-validation and Optuna."""
    
    def __init__(
        self,
        n_trials: int = 100,
        n_folds: int = 5,
        random_state: int = 42,
        study_name: str = "model_optimization",
        storage_dir: str = "models"
    ):
        """
        Initialize the model tuner.
        
        Args:
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            study_name: Name for the optimization study
            storage_dir: Directory to save optimization results
        """
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.study_name = study_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Define scoring metrics
        self.scoring = {
            'mae': make_scorer(mean_absolute_error),
            'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
            'r2': make_scorer(r2_score)
        }
    
    def _get_random_forest_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the hyperparameter search space for Random Forest."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
    
    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        model_class: Any
    ) -> float:
        """Optimization objective function."""
        # Get hyperparameters for this trial
        params = self._get_random_forest_search_space(trial)
        
        # Create and evaluate model with cross-validation
        model = model_class(**params, random_state=self.random_state)
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scores = cross_validate(
            model,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1
        )
        
        # Calculate mean scores
        mae = scores['test_mae'].mean()
        rmse = scores['test_rmse'].mean()
        r2 = scores['test_r2'].mean()
        
        # Log metrics
        trial.set_user_attr('mae', float(mae))
        trial.set_user_attr('rmse', float(rmse))
        trial.set_user_attr('r2', float(r2))
        
        # Return negative MAE as the objective to minimize
        return mae
    
    def tune_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_class: Any
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_class: Model class to tune
            
        Returns:
            Dictionary containing best parameters and cross-validation results
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and run optimization study
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name
        )
        
        study.optimize(
            lambda trial: self._objective(trial, X_scaled, y, model_class),
            n_trials=self.n_trials
        )
        
        # Get best trial results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_metrics = {
            'mae': best_trial.user_attrs['mae'],
            'rmse': best_trial.user_attrs['rmse'],
            'r2': best_trial.user_attrs['r2']
        }
        
        # Save optimization results
        results = {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'n_trials': self.n_trials,
            'n_folds': self.n_folds
        }
        
        results_path = self.storage_dir / f"{self.study_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nOptimization results saved to {results_path}")
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
        
        print("\nCross-validation metrics:")
        for metric, value in best_metrics.items():
            print(f"- {metric}: {value:.4f}")
        
        return results
    
    def load_best_params(self, study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the best parameters from a previous optimization.
        
        Args:
            study_name: Name of the study to load. If None, uses the current study name.
            
        Returns:
            Dictionary containing the best parameters
        """
        study_name = study_name or self.study_name
        results_path = self.storage_dir / f"{study_name}_results.json"
        
        if not results_path.exists():
            raise FileNotFoundError(f"No optimization results found at {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results['best_params'] 