import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from .model_interface import Model
from .random_forest_model import RandomForestModel
from .gradient_boost_model import GradientBoostModel

class ModelEnsemble:
    """Framework for comparing and combining multiple models."""
    
    def __init__(
        self,
        models: Optional[List[Model]] = None,
        weights: Optional[List[float]] = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize the model ensemble.
        
        Args:
            models: List of model instances to use
            weights: List of weights for each model (must sum to 1)
            output_dir: Directory for all outputs (models, predictions, comparisons)
        """
        try:
            # Initialize default models if none provided
            self.models = models or [
                RandomForestModel(),
                GradientBoostModel()
            ]
            
            # Validate and normalize weights
            if weights:
                if len(weights) != len(self.models):
                    raise ValueError("Number of weights must match number of models")
                if abs(sum(weights) - 1.0) > 1e-6:
                    raise ValueError("Weights must sum to 1")
                self.weights = weights
            else:
                # Equal weights by default
                self.weights = [1.0 / len(self.models)] * len(self.models)
            
            # Setup consolidated output directory structure
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories with clear purposes
            self.current_dir = self.output_dir / "current"  # Current model state
            self.initial_dir = self.output_dir / "initial"  # Initial training data
            self.viz_dir = self.output_dir / "visualizations"  # Visualizations
            
            for dir_path in [self.current_dir, self.initial_dir, self.viz_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initialized model ensemble with {len(self.models)} models")
            logger.info(f"Output directory structure created at {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model ensemble: {str(e)}", exc_info=True)
            raise
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp for file naming."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return self._convert_numpy_types(obj.tolist())
        return obj
    
    def _save_comparison_metrics(self, comparison_data: Dict[str, Any], timestamp: str) -> Path:
        """Save comparison metrics to JSON file."""
        try:
            # Convert numpy types to native Python types
            serializable_data = self._convert_numpy_types(comparison_data)
            
            metrics_file = self.comparisons_dir / f"comparison_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            logger.info(f"Saved comparison metrics to {metrics_file}")
            return metrics_file
        except Exception as e:
            logger.error(f"Failed to save comparison metrics: {str(e)}", exc_info=True)
            raise
    
    def _save_comparison_plot(self, comparison_data: Dict[str, Any], viz_file: Path) -> None:
        """Generate and save comparison visualization."""
        try:
            # Use default style with grid
            plt.style.use('default')
            
            # Create figure with consistent styling
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Model Comparison Metrics', fontsize=14, y=1.05)
            
            # Set common style elements
            bar_color = '#2ecc71'
            grid_color = '#ecf0f1'
            text_color = '#2c3e50'
            
            # Extract model names and metrics
            model_names = list(comparison_data['models'].keys())
            mae_values = [comparison_data['models'][m]['metrics']['mae'] for m in model_names]
            r2_values = [comparison_data['models'][m]['metrics']['r2'] for m in model_names]
            roi_values = [comparison_data['models'][m]['metrics'].get('average_roi', 0) for m in model_names]
            
            # Helper function for consistent bar plots
            def style_axis(ax, title, ylabel):
                ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
                ax.set_title(title, fontsize=12, pad=10, color=text_color)
                ax.set_ylabel(ylabel, fontsize=10, color=text_color)
                ax.tick_params(axis='both', colors=text_color)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                for spine in ax.spines.values():
                    spine.set_color(grid_color)
            
            # MAE comparison
            sns.barplot(x=model_names, y=mae_values, ax=ax1, color=bar_color)
            style_axis(ax1, 'Mean Absolute Error', 'MAE ($)')
            
            # R² comparison
            sns.barplot(x=model_names, y=r2_values, ax=ax2, color=bar_color)
            style_axis(ax2, 'R² Score', 'R²')
            
            # ROI comparison
            sns.barplot(x=model_names, y=roi_values, ax=ax3, color=bar_color)
            style_axis(ax3, 'Average ROI', 'ROI (%)')
            
            # Add value labels on top of bars
            for ax in [ax1, ax2, ax3]:
                for i, p in enumerate(ax.patches):
                    value = p.get_height()
                    ax.annotate(f'{value:.2f}',
                              (p.get_x() + p.get_width() / 2., value),
                              ha='center', va='bottom',
                              color=text_color,
                              fontsize=9)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot with high quality
            plt.savefig(viz_file, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            plt.close()
            
            logger.info(f"Saved comparison visualization to {viz_file}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison plot: {str(e)}", exc_info=True)
            raise 
    
    def _save_model_states(self, timestamp: str) -> Path:
        """Save current state of all models."""
        try:
            model_states = {}
            model_dir = self.models_dir / timestamp
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for i, model in enumerate(self.models):
                model_name = type(model).__name__
                model_path = model_dir / f"{model_name.lower()}.joblib"
                model.save(str(model_path))
                model_states[model_name] = str(model_path)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'model_paths': model_states,
                'weights': self.weights
            }
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved model states to {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to save model states: {str(e)}", exc_info=True)
            raise
    
    def _cleanup_old_outputs(self) -> None:
        """Clean up current outputs before new run."""
        try:
            # Clear current directory
            if self.current_dir.exists():
                for item in self.current_dir.glob("**/*"):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        for subitem in item.glob("*"):
                            subitem.unlink()
                        item.rmdir()
                logger.debug("Cleared current outputs")
            
            # Remove old visualization
            old_viz = self.viz_dir / "model_comparison.png"
            if old_viz.exists():
                old_viz.unlink()
                logger.debug("Removed old visualization")
                    
        except Exception as e:
            logger.warning(f"Failed to clean up outputs: {str(e)}")
            # Continue execution even if cleanup fails
    
    def compare_models(
        self,
        data: pd.DataFrame,
        tune_first: bool = False,
        n_trials: int = 100,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Compare all models in the ensemble.
        
        Args:
            data: Training data
            tune_first: Whether to tune models before comparison
            n_trials: Number of optimization trials if tuning
            n_folds: Number of cross-validation folds if tuning
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Clean up old outputs first
            self._cleanup_old_outputs()
            
            logger.info("Starting model comparison...")
            comparison = {'models': {}}
            
            # Train and evaluate each model
            for model in self.models:
                model_name = type(model).__name__
                logger.info(f"\nEvaluating {model_name}...")
                
                # Train model
                metrics = model.train(
                    data,
                    tune_first=tune_first,
                    n_trials=n_trials,
                    n_folds=n_folds
                )
                
                # Get feature importance
                importance = model.get_feature_importance()
                
                # Convert numpy types to native Python types
                metrics = self._convert_numpy_types(metrics)
                importance = self._convert_numpy_types(importance)
                
                comparison['models'][model_name] = {
                    'metrics': metrics,
                    'feature_importance': importance
                }
            
            # Determine best model based on MAE
            best_model = min(
                comparison['models'].items(),
                key=lambda x: x[1]['metrics']['mae']
            )[0]
            comparison['best_model'] = best_model
            
            # Calculate optimal weights based on inverse MAE
            mae_values = [m['metrics']['mae'] for m in comparison['models'].values()]
            inv_mae = [1/mae for mae in mae_values]
            sum_inv_mae = sum(inv_mae)
            self.weights = [float(w/sum_inv_mae) for w in inv_mae]  # Convert to native float
            comparison['ensemble_weights'] = dict(zip(
                comparison['models'].keys(),
                self.weights
            ))
            
            # Save all outputs
            outputs = self._save_run_outputs(
                predictions=pd.Series(),  # Empty series as no predictions yet
                prediction_stats=[],      # Empty stats as no predictions yet
                comparison_data=comparison,
                initial_data=data         # Save initial data for future comparison
            )
            
            comparison['artifacts'] = {
                path_type: str(path) for path_type, path in outputs.items()
            }
            
            logger.info(f"\nComparison complete. Best model: {best_model}")
            logger.info(f"Results saved to {self.current_dir}")
            
            return self._convert_numpy_types(comparison)  # Ensure return value is serializable
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}", exc_info=True)
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data for predictions."""
        if data.empty:
            raise ValueError("Empty input data")
            
        # Check required columns
        required_columns = [
            'price', 'weight', 'review_rating', 'review_count',
            'competitors', 'estimated_monthly_sales', 'fba_fees', 'cogs'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for NaN values
        nan_columns = data[required_columns].columns[data[required_columns].isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"Found NaN values in columns: {nan_columns}")
            
        # Validate numeric types
        non_numeric = []
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                non_numeric.append(col)
        if non_numeric:
            raise ValueError(f"Non-numeric data in columns: {non_numeric}")
            
        # Validate value ranges
        validations = {
            'price': (0, 10000),
            'weight': (0, 1000),
            'review_rating': (0, 5),
            'review_count': (0, 1000000),
            'competitors': (0, 1000),
            'estimated_monthly_sales': (0, 100000),
            'fba_fees': (0, 1000),
            'cogs': (0, 10000)
        }
        
        for col, (min_val, max_val) in validations.items():
            invalid_values = data[col][(data[col] < min_val) | (data[col] > max_val)]
            if not invalid_values.empty:
                logger.warning(f"Found {len(invalid_values)} values outside expected range [{min_val}, {max_val}] in {col}")

    def _save_predictions(self, predictions: pd.Series, prediction_stats: List[Dict], timestamp: str) -> Path:
        """Save predictions and their metadata to a single location."""
        try:
            # Create predictions directory for this run
            pred_dir = self.predictions_dir / timestamp
            pred_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            predictions_file = pred_dir / "predictions.csv"
            predictions.to_csv(predictions_file)
            
            # Save prediction metadata
            metadata = {
                'timestamp': timestamp,
                'model_stats': prediction_stats,
                'summary_stats': {
                    'min': float(predictions.min()),
                    'max': float(predictions.max()),
                    'mean': float(predictions.mean()),
                    'median': float(predictions.median()),
                    'std': float(predictions.std())
                },
                'model_weights': {
                    type(model).__name__: float(weight)
                    for model, weight in zip(self.models, self.weights)
                }
            }
            
            metadata_file = pred_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved predictions to {pred_dir}")
            return pred_dir
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}", exc_info=True)
            raise

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using weighted ensemble.
        
        Args:
            data: DataFrame containing features for prediction
            
        Returns:
            Series containing weighted average predictions
        """
        try:
            # Clean up old outputs first
            self._cleanup_old_outputs()
            
            # Validate input data
            logger.debug("Validating input data...")
            self._validate_data(data)
            
            predictions = []
            prediction_stats = []
            
            # Get predictions from each model with validation
            for model, weight in zip(self.models, self.weights):
                model_name = type(model).__name__
                logger.debug(f"Getting predictions from {model_name}...")
                
                try:
                    # Preprocess data for this model
                    if hasattr(model, '_preprocess_data'):
                        model_data = model._preprocess_data(data)
                    else:
                        model_data = data
                    
                    # Get predictions
                    model_pred = model.predict(model_data)
                    
                    # Ensure predictions are a pandas Series
                    if isinstance(model_pred, np.ndarray):
                        model_pred = pd.Series(model_pred, index=data.index)
                    
                    # Validate individual model predictions
                    if model_pred.isna().any():
                        nan_count = model_pred.isna().sum()
                        logger.error(f"{model_name} produced {nan_count} NaN predictions")
                        raise ValueError(f"{model_name} produced NaN predictions")
                    
                    # Collect prediction statistics
                    stats = {
                        'model': model_name,
                        'min': float(model_pred.min()),
                        'max': float(model_pred.max()),
                        'mean': float(model_pred.mean()),
                        'std': float(model_pred.std()),
                        'weight': float(weight)
                    }
                    prediction_stats.append(stats)
                    
                    # Apply weight and validate
                    weighted_pred = model_pred.multiply(weight)  # Use pandas multiply
                    if weighted_pred.isna().any():
                        raise ValueError(f"Weighted predictions from {model_name} contain NaN values")
                    
                    predictions.append(weighted_pred)
                    logger.debug(f"{model_name} predictions - Range: ${float(model_pred.min()):.2f} to ${float(model_pred.max()):.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to get predictions from {model_name}: {str(e)}", exc_info=True)
                    raise
            
            if not predictions:
                raise ValueError("No valid predictions from any model")
            
            # Combine predictions using pandas add
            ensemble_pred = predictions[0]
            for pred in predictions[1:]:
                ensemble_pred = ensemble_pred.add(pred, fill_value=0)
            
            # Validate final predictions
            if ensemble_pred.isna().any():
                nan_count = ensemble_pred.isna().sum()
                total_count = len(ensemble_pred)
                logger.error("Prediction statistics before ensemble:")
                for stats in prediction_stats:
                    logger.error(f"- {stats['model']} (weight={stats['weight']:.3f}):")
                    logger.error(f"  Range: ${stats['min']:.2f} to ${stats['max']:.2f}")
                    logger.error(f"  Mean: ${stats['mean']:.2f}, Std: ${stats['std']:.2f}")
                raise ValueError(f"Ensemble produced {nan_count}/{total_count} NaN predictions")
            
            # Validate prediction range
            if ensemble_pred.min() < -10000 or ensemble_pred.max() > 10000:
                logger.warning("Ensemble predictions outside expected range")
                logger.warning("Individual model contributions:")
                for stats in prediction_stats:
                    logger.warning(f"- {stats['model']} (weight={stats['weight']:.3f}):")
                    logger.warning(f"  Range: ${stats['min']:.2f} to ${stats['max']:.2f}")
            
            # Save predictions and metadata
            timestamp = self._get_timestamp()
            self._save_run_outputs(ensemble_pred, prediction_stats, timestamp=timestamp)
            
            # Log prediction statistics
            logger.info(f"Ensemble prediction summary:")
            logger.info(f"- Range: ${float(ensemble_pred.min()):.2f} to ${float(ensemble_pred.max()):.2f}")
            logger.info(f"- Mean: ${float(ensemble_pred.mean()):.2f}")
            logger.info(f"- Median: ${float(ensemble_pred.median()):.2f}")
            logger.info(f"- Std Dev: ${float(ensemble_pred.std()):.2f}")
            
            # Add model weights to log
            logger.info("Model weights used:")
            for model, weight in zip(self.models, self.weights):
                logger.info(f"- {type(model).__name__}: {weight:.3f}")
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}", exc_info=True)
            raise
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the ensemble configuration and models.
        
        Args:
            path: Directory to save ensemble. If None, uses default path.
        """
        try:
            # Clean up old saves if using default path
            if path is None:
                self._cleanup_old_outputs()
            
            timestamp = self._get_timestamp()
            save_dir = Path(path) if path else self.models_dir / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each model
            model_paths = {}
            for i, model in enumerate(self.models):
                model_name = type(model).__name__
                model_path = save_dir / f"{model_name.lower()}.joblib"
                model.save(str(model_path))
                model_paths[model_name] = str(model_path)
            
            # Save ensemble metadata
            metadata = {
                'timestamp': timestamp,
                'model_paths': model_paths,
                'weights': self.weights,
                'model_types': [type(m).__name__ for m in self.models]
            }
            
            metadata_path = save_dir / "ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved ensemble to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {str(e)}", exc_info=True)
            raise
    
    def load(self, path: str) -> None:
        """
        Load ensemble configuration and models.
        
        Args:
            path: Path to saved ensemble directory
        """
        try:
            load_dir = Path(path)
            if not load_dir.exists():
                raise FileNotFoundError(f"No saved ensemble found at {load_dir}")
            
            # Load metadata
            metadata_path = load_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load models
            self.models = []
            models_dir = load_dir / "models"
            if not models_dir.exists():
                raise FileNotFoundError(f"No models directory found at {models_dir}")
                
            for model_type, model_path in metadata['model_paths'].items():
                if model_type == "RandomForestModel":
                    model = RandomForestModel()
                elif model_type == "GradientBoostModel":
                    model = GradientBoostModel()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                model_path = models_dir / f"{model_type.lower()}.joblib"
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                model.load(str(model_path))
                self.models.append(model)
            
            # Load weights
            self.weights = metadata['weights']
            
            logger.info(f"Loaded ensemble from {load_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {str(e)}", exc_info=True)
            raise
    
    def _save_run_outputs(
        self,
        predictions: pd.Series,
        prediction_stats: List[Dict],
        comparison_data: Optional[Dict] = None,
        initial_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """Save all outputs for the current run."""
        try:
            # Save initial data if provided
            if initial_data is not None:
                self._save_initial_data(initial_data)
            
            # Clear current directory
            if self.current_dir.exists():
                for item in self.current_dir.glob("**/*"):
                    if item.is_file():
                        item.unlink()
            
            outputs = {}
            
            # Save predictions and metadata
            if not predictions.empty:
                predictions_file = self.current_dir / "predictions.csv"
                predictions.to_csv(predictions_file)
                outputs['predictions'] = predictions_file
            
            # Save prediction metadata
            if prediction_stats:
                metadata = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'model_stats': prediction_stats,
                    'summary_stats': {
                        'min': float(predictions.min()),
                        'max': float(predictions.max()),
                        'mean': float(predictions.mean()),
                        'median': float(predictions.median()),
                        'std': float(predictions.std())
                    },
                    'model_weights': {
                        type(model).__name__: float(weight)
                        for model, weight in zip(self.models, self.weights)
                    }
                }
                
                metadata_file = self.current_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                outputs['metadata'] = metadata_file
            
            # Save model states
            models_dir = self.current_dir / "models"
            models_dir.mkdir(exist_ok=True)
            model_paths = {}
            
            for model in self.models:
                model_name = type(model).__name__
                model_path = models_dir / f"{model_name.lower()}.joblib"
                model.save(str(model_path))
                model_paths[model_name] = str(model_path)
            
            outputs['models'] = models_dir
            
            # Save comparison data if provided
            if comparison_data:
                comparison_file = self.current_dir / "comparison.json"
                with open(comparison_file, 'w') as f:
                    json.dump(self._convert_numpy_types(comparison_data), f, indent=4)
                outputs['comparison'] = comparison_file
                
                # Save visualization
                viz_file = self.viz_dir / "model_comparison.png"
                self._save_comparison_plot(comparison_data, viz_file)
                outputs['visualization'] = viz_file
            
            logger.info(f"Saved run outputs to {self.current_dir}")
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to save run outputs: {str(e)}", exc_info=True)
            raise
    
    def _save_comparison_plot(self, comparison_data: Dict[str, Any], viz_file: Path) -> None:
        """Generate and save comparison visualization."""
        try:
            # Use default style with grid
            plt.style.use('default')
            
            # Create figure with consistent styling
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Model Comparison Metrics', fontsize=14, y=1.05)
            
            # Set common style elements
            bar_color = '#2ecc71'
            grid_color = '#ecf0f1'
            text_color = '#2c3e50'
            
            # Extract model names and metrics
            model_names = list(comparison_data['models'].keys())
            mae_values = [comparison_data['models'][m]['metrics']['mae'] for m in model_names]
            r2_values = [comparison_data['models'][m]['metrics']['r2'] for m in model_names]
            roi_values = [comparison_data['models'][m]['metrics'].get('average_roi', 0) for m in model_names]
            
            # Helper function for consistent bar plots
            def style_axis(ax, title, ylabel):
                ax.grid(True, linestyle='--', alpha=0.7, color=grid_color)
                ax.set_title(title, fontsize=12, pad=10, color=text_color)
                ax.set_ylabel(ylabel, fontsize=10, color=text_color)
                ax.tick_params(axis='both', colors=text_color)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                for spine in ax.spines.values():
                    spine.set_color(grid_color)
            
            # MAE comparison
            sns.barplot(x=model_names, y=mae_values, ax=ax1, color=bar_color)
            style_axis(ax1, 'Mean Absolute Error', 'MAE ($)')
            
            # R² comparison
            sns.barplot(x=model_names, y=r2_values, ax=ax2, color=bar_color)
            style_axis(ax2, 'R² Score', 'R²')
            
            # ROI comparison
            sns.barplot(x=model_names, y=roi_values, ax=ax3, color=bar_color)
            style_axis(ax3, 'Average ROI', 'ROI (%)')
            
            # Add value labels on top of bars
            for ax in [ax1, ax2, ax3]:
                for i, p in enumerate(ax.patches):
                    value = p.get_height()
                    ax.annotate(f'{value:.2f}',
                              (p.get_x() + p.get_width() / 2., value),
                              ha='center', va='bottom',
                              color=text_color,
                              fontsize=9)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot with high quality
            plt.savefig(viz_file, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            plt.close()
            
            logger.info(f"Saved comparison visualization to {viz_file}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison plot: {str(e)}", exc_info=True)
            raise 