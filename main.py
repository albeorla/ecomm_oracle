import click
from pathlib import Path
from loguru import logger
import json
import shutil
import numpy as np

from src.models.random_forest_model import RandomForestModel
from src.data.mock_data_source import MockDataSource
from src.data.product_specs import ProductSpecs

# Define constants for directory paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

class PredictionPipeline:
    def __init__(self, product_type: str, samples: int, model_dir: str):
        self.product_type = product_type
        self.samples = samples
        self.model_dir = model_dir
        self.model = RandomForestModel(model_dir=model_dir)
        self.data_source = MockDataSource(product_type, n_samples=samples)
        
    def train(self, tune: bool = False):
        """Train the model with current data"""
        train_data = self.data_source.get_training_data()
        
        # Validate data
        if not self.data_source.validate_schema(train_data):
            raise ValueError("Training data failed schema validation")
        
        if not self.data_source.validate_features(train_data):
            raise ValueError("Training data failed feature validation")
        
        # Train model
        metrics = self.model.train(train_data, tune_first=tune)
        
        # Log training metrics
        logger.info("Model Training Metrics:")
        logger.info(f"Mean Absolute Error: ${metrics['mae']:.2f}")
        logger.info(f"R² Score: {metrics['r2']:.3f}")
        logger.info(f"RMSE: ${metrics['rmse']:.2f}")
        
        return metrics
    
    def predict(self):
        """Make predictions using trained model"""
        # Get validation data for predictions
        val_data = self.data_source.get_validation_data()
        
        # Validate data
        if not self.data_source.validate_schema(val_data):
            raise ValueError("Validation data failed schema validation")
            
        if not self.data_source.validate_features(val_data):
            raise ValueError("Validation data failed feature validation")
        
        # Make predictions
        predictions = self.model.predict(val_data)
        
        # Add predictions to validation data (using proper pandas indexing)
        val_data.loc[:, 'predicted_profit'] = predictions
        
        # Calculate comprehensive metrics
        metrics = {
            'product_type': self.product_type,
            'total_products': len(val_data),
            'average_predicted_profit': float(np.mean(predictions)),
            'median_predicted_profit': float(np.median(predictions)),
            'profit_range': [float(np.min(predictions)), float(np.max(predictions))]
        }
        
        # Calculate and log business metrics
        business_metrics = self.model.calculate_business_metrics(val_data, predictions)
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
        
        return val_data, predictions, metrics

def cleanup_old_runs():
    """Clean up output from previous runs."""
    # Clean up output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Recreate necessary directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging() -> None:
    """Configure logging with separate files for general logs and errors."""
    # Add handler for general logs (INFO and above)
    logger.add(
        LOGS_DIR / "app.log",
        format="{time} | {level:<8} | {module}:{function}:{line} - {message}",
        level="INFO",
        rotation=None,
        mode="w"  # Overwrite file on each run
    )
    
    # Add handler for error logs
    logger.add(
        LOGS_DIR / "error.log",
        format="{time} | {level:<8} | {module}:{function}:{line} - {message}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        rotation=None,
        mode="w"  # Overwrite file on each run
    )

def save_predictions(val_data, metrics, product_type: str) -> None:
    """Save predictions and metrics to files.
    
    Args:
        val_data (pd.DataFrame): Validation data with predictions
        metrics (dict): Calculated metrics
        product_type (str): Type of product analyzed
    """
    # Create output DataFrame with predictions
    output_df = val_data.copy()
    
    # Add ASIN scores
    output_df['asin_score'] = output_df.apply(score_asin, axis=1)
    
    # Sort by ASIN score and predicted profit
    output_df = output_df.sort_values(['asin_score', 'predicted_profit'], ascending=[False, False])
    
    # Calculate ROI
    output_df['roi'] = output_df['monthly_profit'] / (output_df['cogs'] * output_df['estimated_monthly_sales'])
    
    # Save detailed predictions
    predictions_file = PREDICTIONS_DIR / f"{product_type}_predictions.csv"
    output_df.to_csv(predictions_file, index=False)
    
    # Add category viability metrics
    category_metrics = analyze_category_viability(output_df)
    metrics.update(category_metrics)
    
    # Save summary metrics
    summary_file = PREDICTIONS_DIR / f"{product_type}_summary.json"
    with open(summary_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.int64, np.int32)):
                clean_metrics[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                clean_metrics[k] = float(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.float64, np.float32)):
                clean_metrics[k] = [float(x) for x in v]
            else:
                clean_metrics[k] = v
        
        json.dump(clean_metrics, f, indent=4)
        
    logger.info(f"Results saved to {PREDICTIONS_DIR}")
    
    # Log category viability summary
    logger.info("\nCategory Viability Summary:")
    logger.info(f"Total Products Analyzed: {category_metrics['total_products']}")
    logger.info(f"Profitable Products: {category_metrics['profitable_percentage']}")
    logger.info(f"Products with High ROI: {category_metrics['high_roi_count']}")
    logger.info(f"Products with Low Competition: {category_metrics['low_competition']}")
    logger.info(f"Products with Good BSR: {category_metrics['good_bsr']}")
    logger.info(f"Average Monthly Profit: ${category_metrics['avg_monthly_profit']:.2f}")
    logger.info(f"Top 10 Average Profit: ${category_metrics['top_10_avg_profit']:.2f}")
    
    # Log top opportunities
    logger.info("\nTop 5 Opportunities:")
    top_5 = output_df.head(5)
    for _, row in top_5.iterrows():
        logger.info(
            f"ASIN: {row['asin']} | "
            f"Score: {row['asin_score']}/15 | "
            f"Profit: ${row['monthly_profit']:.2f} | "
            f"ROI: {row['roi']*100:.1f}% | "
            f"BSR: {row['bsr']:,} | "
            f"Competitors: {row['competitors']}"
        )

def analyze_product_type(
    product_type: str,
    samples: int,
    tune: bool,
    predict_only: bool
) -> None:
    """Analyze profitability for a product type."""
    try:
        pipeline = PredictionPipeline(
            product_type=product_type,
            samples=samples,
            model_dir=str(MODELS_DIR)
        )
        
        if not predict_only:
            pipeline.train(tune=tune)
        
        val_data, predictions, metrics = pipeline.predict()
        
        # Save all results
        save_predictions(val_data, metrics, product_type)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

def analyze_category_viability(df):
    """Analyze overall category viability.
    
    Args:
        df (pd.DataFrame): Product data
        
    Returns:
        dict: Category viability metrics
    """
    metrics = {
        'total_products': len(df),
        'profitable_products': len(df[df['monthly_profit'] > 0]),
        'profitable_percentage': f"{(len(df[df['monthly_profit'] > 0]) / len(df) * 100):.1f}%",
        'high_roi_count': len(df[df['predicted_profit'] / (df['cogs'] * df['estimated_monthly_sales']) > 0.4]),
        'low_competition': len(df[df['competitors'] < 10]),
        'good_bsr': len(df[df['bsr'] < 50000]),
        'avg_monthly_profit': df['monthly_profit'].mean(),
        'median_monthly_profit': df['monthly_profit'].median(),
        'profit_std': df['monthly_profit'].std(),  # Measure of risk/volatility
        'top_10_avg_profit': df.nlargest(10, 'monthly_profit')['monthly_profit'].mean()
    }
    return metrics

def score_asin(row):
    """Score an individual ASIN based on key metrics.
    
    Args:
        row: DataFrame row containing ASIN data
        
    Returns:
        float: Score from 0-15
    """
    score = 0
    
    # Profitability (0-5 points)
    monthly_profit = row['monthly_profit']
    if monthly_profit > 0: score += 1
    if monthly_profit > 100: score += 1
    if monthly_profit > 300: score += 1
    roi = monthly_profit / (row['cogs'] * row['estimated_monthly_sales'])
    if roi > 0.4: score += 1
    if roi > 0.6: score += 1
    
    # Competition (0-3 points)
    if row['competitors'] < 15: score += 1
    if row['competitors'] < 10: score += 1
    if row['seller_count'] < 5: score += 1
    
    # Demand (0-4 points)
    if row['bsr'] < 100000: score += 1
    if row['bsr'] < 50000: score += 1
    if row['estimated_monthly_sales'] > 10: score += 1
    if row['estimated_monthly_sales'] > 20: score += 1
    
    # Reviews (0-3 points)
    if row['review_rating'] > 4.0: score += 1
    if row['review_count'] > 10: score += 1
    if row['review_velocity'] > 0: score += 1
    
    return score

@click.group()
def cli():
    """Amazon FBA Product Profitability Analyzer"""
    cleanup_old_runs()
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
@click.option('--tune/--no-tune', default=False, help='Tune model hyperparameters')
@click.option('--predict-only', is_flag=True, help='Only make predictions using existing model')
def analyze(
    product_type: str,
    samples: int,
    tune: bool,
    predict_only: bool
):
    """Analyze profitability for a product type."""
    try:
        analyze_product_type(
            product_type=product_type,
            samples=samples,
            tune=tune,
            predict_only=predict_only
        )
    except Exception as e:
        logger.error("Fatal error occurred")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()