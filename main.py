from src.data.csv_data_source import CSVDataSource
from src.models.random_forest_model import RandomForestModel
from src.data.product_specs import ProductSpecs
from src.utils.logger import logger, log_execution_time
import click
from pathlib import Path
import sys
import json

def format_currency(value: float) -> str:
    """Format a value as currency."""
    return f"${value:,.2f}"

def analyze_product_type(
    product_type: str,
    n_samples: int = 1000,
    model_dir: str = "models",
    data_dir: str = "data",
    tune_model: bool = False,
    predict_only: bool = False,
    output_dir: str = "predictions"
) -> None:
    """
    Analyze profitability for a specific product type.
    
    Args:
        product_type: Type of product to analyze
        n_samples: Number of samples to generate
        model_dir: Directory for model files
        data_dir: Directory for data files
        tune_model: Whether to tune hyperparameters
        predict_only: Whether to only make predictions using existing model
        output_dir: Directory for prediction outputs
    """
    try:
        # Validate product type
        if not ProductSpecs.is_valid_product_type(product_type):
            available_types = ProductSpecs.get_available_types()
            logger.error(f"Invalid product type: {product_type}")
            logger.info(f"Available product types: {', '.join(available_types)}")
            sys.exit(1)

        # Set up paths
        data_path = Path(data_dir) / f"{product_type}_products.csv"
        model_path = Path(model_dir) / f"{product_type}_model.joblib"
        
        with log_execution_time("Data loading"):
            # Initialize data source and load data
            data_source = CSVDataSource(
                file_path=str(data_path),
                n_samples=n_samples,
                product_type=product_type  # Explicitly pass product_type
            )
            data = data_source.get_products()
            
            # Log basic statistics
            stats = {
                'total_products': len(data),
                'price_range': (data['price'].min(), data['price'].max()),
                'avg_price': data['price'].mean(),
                'avg_profit': data['monthly_profit'].mean(),
                'median_profit': data['monthly_profit'].median()
            }
            logger.info(f"Loaded {stats['total_products']} products")
            logger.info(f"Price Range: {format_currency(stats['price_range'][0])} - {format_currency(stats['price_range'][1])}")
            logger.info(f"Average Price: {format_currency(stats['avg_price'])}")
            logger.info(f"Average Profit: {format_currency(stats['avg_profit'])}")
        
        # Initialize model
        model = RandomForestModel(model_dir=model_dir)
        
        if not predict_only:
            # Train new model
            with log_execution_time("Model training"):
                if tune_model:
                    logger.info("Tuning hyperparameters...")
                    model.tune(data)
                
                metrics = model.train(data)
                logger.info(f"Training metrics: MAE={format_currency(metrics['mae'])}, R²={metrics['r2']:.3f}")
        else:
            # Load existing model
            with log_execution_time("Model loading"):
                try:
                    model.load(model_path)
                except FileNotFoundError:
                    logger.error(f"No trained model found at {model_path}")
                    logger.info("Please train a model first (remove --predict-only flag)")
                    sys.exit(1)
        
        # Make predictions
        with log_execution_time("Prediction generation"):
            predictions = model.predict(data)
            
            # Add predictions to data
            results = data.copy()
            results['predicted_profit'] = predictions
            
            if 'monthly_profit' in results.columns:
                results['profit_difference'] = results['predicted_profit'] - results['monthly_profit']
                results['prediction_accuracy'] = (
                    1 - abs(results['profit_difference']) / results['monthly_profit']
                ).clip(0, 1)
            
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed predictions
            results.to_csv(output_path / f"{product_type}_predictions.csv", index=False)
            
            # Generate and save summary
            summary = {
                'product_type': product_type,
                'total_products': len(results),
                'average_predicted_profit': float(predictions.mean()),
                'median_predicted_profit': float(predictions.median()),
                'profit_range': (float(predictions.min()), float(predictions.max()))
            }
            
            if 'monthly_profit' in results.columns:
                summary.update({
                    'mean_absolute_error': float(abs(results['profit_difference']).mean()),
                    'prediction_accuracy': float(results['prediction_accuracy'].mean()),
                    'predictions_within_10_percent': float(
                        (results['prediction_accuracy'] >= 0.9).mean()
                    )
                })
            
            with open(output_path / f"{product_type}_summary.json", 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Log summary statistics
            logger.info(f"\nPrediction Summary for {product_type}:")
            logger.info(f"Total products analyzed: {summary['total_products']:,}")
            logger.info(f"Average predicted profit: {format_currency(summary['average_predicted_profit'])}")
            logger.info(f"Profit range: {format_currency(summary['profit_range'][0])} to {format_currency(summary['profit_range'][1])}")
            
            if 'mean_absolute_error' in summary:
                logger.info(f"Prediction accuracy: {summary['prediction_accuracy']:.1%}")
                logger.info(f"Predictions within 10%: {summary['predictions_within_10_percent']:.1%}")
            
            logger.success(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        raise

def validate_product_type(ctx, param, value):
    """Validate that the product type exists"""
    if not ProductSpecs.is_valid_product_type(value):
        available_types = ProductSpecs.get_available_types()
        raise click.BadParameter(
            f"Invalid product type: {value}\nAvailable types: {', '.join(available_types)}"
        )
    return value

@click.group()
def cli():
    """Amazon FBA Product Profitability Analyzer

    Analyze and predict profitability for various Amazon FBA product types.
    """
    pass

@cli.command()
def list_products():
    """List all available product types"""
    available_types = ProductSpecs.get_available_types()
    logger.info(f"Available product types ({len(available_types)}):")
    for product_type in available_types:
        logger.info(f"- {product_type}")

@cli.command()
@click.argument('product_type', type=str, callback=validate_product_type)
@click.option('--samples', '-n', type=click.IntRange(min=1), default=1000, show_default=True,
              help='Number of samples to generate')
@click.option('--model-dir', type=click.Path(), default='models', show_default=True,
              help='Directory for model files')
@click.option('--data-dir', type=click.Path(), default='data', show_default=True,
              help='Directory for data files')
@click.option('--output-dir', type=click.Path(), default='predictions', show_default=True,
              help='Directory for prediction outputs')
@click.option('--tune/--no-tune', default=False, show_default=True,
              help='Tune model hyperparameters')
@click.option('--predict-only', is_flag=True,
              help='Only make predictions using existing model')
def analyze(product_type, samples, model_dir, data_dir, output_dir, tune, predict_only):
    """Analyze profitability for a specific product type"""
    logger.info("Starting Amazon FBA Product Profitability Analyzer")
    try:
        analyze_product_type(
            product_type=product_type,
            n_samples=samples,
            model_dir=model_dir,
            data_dir=data_dir,
            tune_model=tune,
            predict_only=predict_only,
            output_dir=output_dir
        )
    except Exception as e:
        logger.exception("Fatal error occurred")
        sys.exit(1)

if __name__ == "__main__":
    cli()