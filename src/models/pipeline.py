from typing import Optional, Dict, Any
import mlflow
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from models.base_model import BaseModel
from models.model_tuner import ModelTuner

class MLPipeline:
    """Standard ML pipeline for iterative model training and evaluation."""
    
    def __init__(
        self,
        model: BaseModel,
        experiment_name: str = "product_profitability",
        run_name: Optional[str] = None,
        artifacts_dir: str = "models/artifacts"
    ):
        self.model = model
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Set up model registry
        self.model_registry_dir = Path("models/registry")
        self.model_registry_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data from CSV."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Basic preprocessing
        # Drop any rows with missing values
        df = df.dropna()
        
        # Ensure required columns exist
        required_columns = {'monthly_profit'}  # Add other required columns
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    def run_pipeline(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        hyperparameter_tune: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline with MLflow tracking.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Optional path to validation data
            hyperparameter_tune: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of metrics and artifacts
        """
        logger.info("Starting pipeline run")
        
        # Load data
        train_data = self._load_data(train_data_path)
        val_data = self._load_data(val_data_path) if val_data_path else None
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            
            # Log git commit if available
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
                mlflow.set_tag("git_commit", repo.head.object.hexsha)
            except Exception as e:
                logger.warning(f"Could not log git commit: {e}")
            
            # Log parameters
            params = {
                "model_type": self.model.__class__.__name__,
                "train_data_path": train_data_path,
                "val_data_path": val_data_path,
                "hyperparameter_tune": hyperparameter_tune,
                "train_samples": len(train_data),
                "val_samples": len(val_data) if val_data is not None else 0,
                **kwargs
            }
            mlflow.log_params(params)
            
            # Hyperparameter tuning
            if hyperparameter_tune:
                logger.info("Running hyperparameter tuning")
                tuner = ModelTuner(self.model)
                best_params = tuner.tune(train_data)
                self.model.set_params(**best_params)
                mlflow.log_params(best_params)
            
            # Train model
            logger.info("Training model")
            train_metrics = self.model.train(train_data)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            
            # Validate model
            if val_data is not None:
                logger.info("Validating model")
                val_metrics = self.model.evaluate(val_data)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Log validation predictions
                val_preds = self.model.predict(val_data)
                pred_df = pd.DataFrame({
                    'actual': val_data['monthly_profit'],
                    'predicted': val_preds
                })
                pred_path = self.artifacts_dir / "validation_predictions.csv"
                pred_df.to_csv(pred_path, index=False)
                mlflow.log_artifact(pred_path)
            
            # Save model artifacts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.artifacts_dir / f"model_{timestamp}.pkl"
            self.model.save(model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                self.model.model,
                "model",
                registered_model_name="product_profitability_model"
            )
            
            # Save to model registry with timestamp
            registry_path = self.model_registry_dir / f"model_{timestamp}.pkl"
            self.model.save(registry_path)
            
            # Generate performance visualizations
            self._log_visualizations(train_data, val_data)
            
            # Log feature importance if available
            if hasattr(self.model.model, 'feature_importances_'):
                self._log_feature_importance(train_data)
            
            return {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics if val_data is not None else None,
                "model_path": str(model_path),
                "run_id": run_id
            }
    
    def _log_visualizations(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
        """Generate and log performance visualizations to MLflow."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prediction vs Actual plot
        plt.figure(figsize=(10, 6))
        train_preds = self.model.predict(train_data)
        plt.scatter(train_data['monthly_profit'], train_preds, alpha=0.5, label='Train')
        if val_data is not None:
            val_preds = self.model.predict(val_data)
            plt.scatter(val_data['monthly_profit'], val_preds, alpha=0.5, label='Validation')
        
        # Add perfect prediction line
        min_val = min(train_data['monthly_profit'].min(), train_preds.min())
        max_val = max(train_data['monthly_profit'].max(), train_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Profit')
        plt.ylabel('Predicted Profit')
        plt.title('Prediction vs Actual')
        plt.legend()
        
        # Save and log plot
        plot_path = self.artifacts_dir / "prediction_scatter.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Residual plot
        plt.figure(figsize=(10, 6))
        residuals = train_preds - train_data['monthly_profit']
        plt.scatter(train_preds, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Profit')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Save and log residual plot
        plot_path = self.artifacts_dir / "residuals.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
    
    def _log_feature_importance(self, train_data: pd.DataFrame):
        """Log feature importance plot if available."""
        import matplotlib.pyplot as plt
        
        # Get feature importance
        importance = self.model.model.feature_importances_
        features = train_data.drop('monthly_profit', axis=1).columns
        
        # Create importance plot
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({'feature': features, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        
        # Save and log plot
        plot_path = self.artifacts_dir / "feature_importance.png"
        plt.savefig(plot_path, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Log feature importance as CSV
        importance_df.to_csv(self.artifacts_dir / "feature_importance.csv", index=False)
        mlflow.log_artifact(self.artifacts_dir / "feature_importance.csv")