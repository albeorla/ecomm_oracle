import click
from pathlib import Path
from loguru import logger
import json

from src.models.random_forest_model import RandomForestModel
from src.data.mock_data_source import MockDataSource
from src.data.product_specs import ProductSpecs

def setup_logging(log_file: str = "logs/error.log") -> None:
    """Configure logging."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path,
        format="{time} | {level}    | {module}:{function}:{line} - {message}",
        level="ERROR",
        backtrace=True,
        diagnose=True
    )

def save_predictions(predictions, metrics, product_type: str, output_dir: str) -> None:
    """Save predictions and metrics to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed predictions
    predictions_file = output_path / f"{product_type}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    
    # Save summary metrics
    summary_file = output_path / f"{product_type}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Results saved to {output_path}")

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
        # Initialize data source
        data_source = MockDataSource(product_type, n_samples=samples)
        
        # Initialize model
        model = RandomForestModel(model_dir=model_dir)
        
        if not predict_only:
            # Get training data
            train_data = data_source.get_training_data()
            
            # Validate data
            if not data_source.validate_schema(train_data):
                raise ValueError("Training data failed schema validation")
            
            if not data_source.validate_features(train_data):
                raise ValueError("Training data failed feature validation")
            
            # Train model
            training_metrics = model.train(train_data, tune_first=tune)
            
            # Log training metrics
            logger.info("Model Training Metrics:")
            logger.info(f"Mean Absolute Error: ${training_metrics['mae']:.2f}")
            logger.info(f"R² Score: {training_metrics['r2']:.3f}")
            logger.info(f"RMSE: ${training_metrics['rmse']:.2f}")
        
        # Get validation data for predictions
        val_data = data_source.get_validation_data()
        
        # Validate data
        if not data_source.validate_schema(val_data):
            raise ValueError("Validation data failed schema validation")
            
        if not data_source.validate_features(val_data):
            raise ValueError("Validation data failed feature validation")
        
        # Make predictions
        predictions = model.predict(val_data)
        
        # Calculate comprehensive metrics
        metrics = {
            'product_type': product_type,
            'total_products': len(val_data),
            'average_predicted_profit': predictions.mean(),
            'median_predicted_profit': predictions.median(),
            'profit_range': [predictions.min(), predictions.max()]
        }
        
        # Calculate and log business metrics
        business_metrics = model.calculate_business_metrics(val_data, predictions)
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
        
        # Save all results
        save_predictions(val_data, metrics, product_type, output_dir)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
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