import click
from pathlib import Path
from loguru import logger
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys

from src.models.model_ensemble import ModelEnsemble
from src.data.mock_data_source import MockDataSource
from src.data.product_specs import ProductSpecs
from src.utils.logger import setup_logger

def setup_logging(log_file: str = "logs/error.log") -> None:
    """Configure logging with separate handlers for different log levels."""
    setup_logger(
        app_log_path="logs/app.log",
        error_log_path=log_file
    )

def save_predictions(
    predictions: pd.Series,
    metrics: dict,
    comparison: dict,
    product_type: str,
    output_dir: str
) -> None:
    """Save predictions, metrics, and model comparison to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed predictions
    predictions_file = output_path / f"{product_type}_predictions.csv"
    predictions.to_csv(predictions_file, index=True)
    
    # Save summary metrics
    summary_file = output_path / f"{product_type}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model comparison
    comparison_file = output_path / f"{product_type}_model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=4)
        
    logger.success(f"Results saved to {output_path}")

def analyze_product_type(
    product_type: str,
    samples: int,
    model_dir: str,
    output_dir: str,
    tune: bool,
    predict_only: bool
) -> None:
    """Analyze profitability for a product type."""
    try:
        logger.info(f"Starting analysis for {product_type} with {samples} samples")
        
        # Initialize data source
        try:
            data_source = MockDataSource(product_type, n_samples=samples)
        except Exception as e:
            logger.error(f"Failed to initialize data source: {str(e)}", exc_info=True)
            raise
        
        # Initialize ensemble
        try:
            ensemble = ModelEnsemble(output_dir=model_dir)
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {str(e)}", exc_info=True)
            raise
        
        if not predict_only:
            # Get training data
            try:
                train_data = data_source.get_training_data()
                logger.debug(f"Got training data with shape: {train_data.shape}")
            except Exception as e:
                logger.error("Failed to get training data", exc_info=True)
                raise
            
            # Validate data
            try:
                if not data_source.validate_schema(train_data):
                    error_msg = "Training data failed schema validation"
                    logger.error(error_msg, extra={"data_info": train_data.info()})
                    raise ValueError(error_msg)
                
                if not data_source.validate_features(train_data):
                    error_msg = "Training data failed feature validation"
                    logger.error(error_msg, extra={"features": list(train_data.columns)})
                    raise ValueError(error_msg)
            except Exception as e:
                logger.error("Data validation failed", exc_info=True)
                raise
            
            # Compare and train models
            try:
                comparison = ensemble.compare_models(
                    data=train_data,
                    tune_first=tune,
                    n_trials=100 if tune else 0,
                    n_folds=5
                )
            except Exception as e:
                logger.error("Model comparison failed", exc_info=True)
                raise
            
            # Log training metrics for each model and ensemble
            logger.info("\nModel Performance Comparison:")
            individual_maes = []
            for model_name, results in comparison['models'].items():
                metrics = results['metrics']
                individual_maes.append(metrics['mae'])
                logger.info(f"\n{model_name}:")
                logger.info(f"Mean Absolute Error: ${metrics['mae']:.2f}")
                logger.info(f"R² Score: {metrics['r2']:.3f}")
                logger.info(f"RMSE: ${metrics['rmse']:.2f}")
            
            # Log ensemble advantage
            best_individual_mae = min(individual_maes)
            ensemble_predictions = ensemble.predict(train_data)
            ensemble_mae = mean_absolute_error(train_data['monthly_profit'], ensemble_predictions)
            mae_improvement = ((best_individual_mae - ensemble_mae) / best_individual_mae) * 100
            
            logger.info(f"\nEnsemble Performance:")
            logger.info(f"Best Individual Model MAE: ${best_individual_mae:.2f}")
            logger.info(f"Ensemble MAE: ${ensemble_mae:.2f}")
            logger.info(f"MAE Improvement: {mae_improvement:.1f}%")
            
            # Log best model and weights
            logger.info(f"\nBest Individual Model: {comparison['best_model']}")
            logger.info("Ensemble Weights:")
            for model, weight in comparison['ensemble_weights'].items():
                logger.info(f"{model}: {weight:.3f}")
        else:
            try:
                ensemble = ModelEnsemble.load(model_dir)
                comparison = None
            except FileNotFoundError:
                logger.error(f"No ensemble found in {model_dir}", exc_info=True)
                raise
            except Exception as e:
                logger.error("Failed to load ensemble", exc_info=True)
                raise
        
        # Get validation data
        try:
            val_data = data_source.get_validation_data()
            logger.debug(f"Got validation data with shape: {val_data.shape}")
        except Exception as e:
            logger.error("Failed to get validation data", exc_info=True)
            raise
        
        # Validate data
        try:
            if not data_source.validate_schema(val_data):
                error_msg = "Validation data failed schema validation"
                logger.error(error_msg, extra={"data_info": val_data.info()})
                raise ValueError(error_msg)
            
            if not data_source.validate_features(val_data):
                error_msg = "Validation data failed feature validation"
                logger.error(error_msg, extra={"features": list(val_data.columns)})
                raise ValueError(error_msg)
        except Exception as e:
            logger.error("Data validation failed", exc_info=True)
            raise
        
        # Make predictions with error bounds checking
        try:
            predictions = ensemble.predict(val_data)
            
            if predictions.isna().any():
                error_msg = "Ensemble produced NaN predictions"
                logger.error(error_msg, extra={"nan_count": predictions.isna().sum()})
                raise ValueError(error_msg)
            
            # Validate prediction ranges with detailed logging
            min_profit = val_data['monthly_profit'].min()
            max_profit = val_data['monthly_profit'].max()
            margin = (max_profit - min_profit) * 0.2
            
            out_of_range = (predictions < min_profit - margin) | (predictions > max_profit + margin)
            if out_of_range.any():
                logger.warning(
                    "Some predictions outside expected range",
                    extra={
                        "out_of_range_count": out_of_range.sum(),
                        "min_prediction": float(predictions.min()),
                        "max_prediction": float(predictions.max()),
                        "expected_range": [float(min_profit - margin), float(max_profit + margin)]
                    }
                )
        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            raise
        
        # Calculate comprehensive metrics
        metrics = {
            'product_type': product_type,
            'total_products': len(val_data),
            'average_predicted_profit': float(predictions.mean()),
            'median_predicted_profit': float(predictions.median()),
            'profit_range': [float(predictions.min()), float(predictions.max())]
        }
        
        # Calculate and log business metrics using ensemble predictions
        business_metrics = ensemble.models[0].calculate_business_metrics(val_data, predictions)
        metrics.update(business_metrics)
        
        # Log business performance metrics
        logger.info("\nBusiness Performance Metrics:")
        logger.info(f"Average ROI: {business_metrics['average_roi']:.1f}%")
        logger.info(f"Average Profit Margin: {business_metrics['average_profit_margin']:.1f}%")
        logger.info(f"Break-even Units: {business_metrics['break_even_units']:.1f}")
        logger.info(f"Average Monthly Sales: {business_metrics['average_monthly_sales']:.1f}")
        logger.info(f"Payback Period: {business_metrics['payback_period_months']:.1f} months")
        logger.info(f"Competition Impact: {business_metrics['competitor_impact']:.3f}")
        logger.info(f"Review Impact: {business_metrics['review_impact']:.3f}")
        
        # Calculate prediction accuracy metrics
        actual_profits = val_data['monthly_profit']
        percent_diff = abs((predictions - actual_profits) / actual_profits)
        within_ten_percent = (percent_diff <= 0.10).mean() * 100
        
        logger.info("\nPrediction Accuracy Metrics:")
        logger.info(f"Predictions within 10%: {within_ten_percent:.1f}%")
        
        # Add ensemble information to metrics
        if comparison:
            metrics['ensemble_info'] = {
                'best_model': comparison['best_model'],
                'model_weights': comparison['ensemble_weights'],
                'ensemble_improvement': float(mae_improvement)
            }
        
        # Save all results
        save_predictions(predictions, metrics, comparison, product_type, output_dir)
        
    except Exception as e:
        logger.error(
            "Analysis failed",
            exc_info=True,
            extra={
                "product_type": product_type,
                "samples": samples,
                "model_dir": model_dir,
                "output_dir": output_dir,
                "tune": tune,
                "predict_only": predict_only
            }
        )
        raise

@click.group()
def cli():
    """Amazon FBA Product Profitability Analyzer"""
    setup_logging()

@cli.command()
def list_products():
    """List available product types."""
    products = ProductSpecs.get_available_products()
    click.echo("\nAvailable product types:")
    for product in products:
        click.echo(f"- {product}")

@cli.command()
@click.argument('product_type')
@click.option('-n', '--samples', default=1000, help='Number of samples to generate')
@click.option('--model-dir', default='models', help='Directory for model files')
@click.option('--data-dir', default='data', help='Directory for data files')
@click.option('--output-dir', default='predictions', help='Directory for prediction outputs')
@click.option('--tune/--no-tune', default=False, help='Tune model hyperparameters')
@click.option('--predict-only', is_flag=True, help='Only make predictions using existing model')
def analyze(
    product_type: str,
    samples: int,
    model_dir: str,
    data_dir: str,
    output_dir: str,
    tune: bool,
    predict_only: bool
):
    """Analyze profitability for a product type."""
    try:
        analyze_product_type(
            product_type=product_type,
            samples=samples,
            model_dir=model_dir,
            output_dir=output_dir,
            tune=tune,
            predict_only=predict_only
        )
    except Exception as e:
        logger.error("Fatal error occurred")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()