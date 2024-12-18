# Amazon FBA Product Profitability Analyzer (Bootstrap PoC)

A proof-of-concept system for evaluating Amazon FBA product profitability using synthetic data. This project provides a bootstrapping framework designed to be replaced with real historical data for actual product performance analysis.

## Purpose

This PoC serves three main goals:
1. **Bootstrap Framework**: Establish the evaluation pipeline and metrics system using synthetic data
2. **System Design**: Create pluggable interfaces ready for real data integration
3. **Hypothesis Testing**: Validate the analysis approach before investing in real data collection

## Current Limitations

As a bootstrapping PoC, this system has important limitations to note:
- Uses synthetic data generated from assumed distributions
- Performance metrics are for system validation only
- Business metrics need validation with real-world data
- Correlations and relationships are simulated

## Transition to Production

To transition this PoC to a production system:

1. **Data Source Replacement**:
   - Replace `product_generator.py` with real data connectors
   - Validate and adjust feature distributions
   - Implement proper data validation
   - Add historical data tracking

2. **Metric Validation**:
   - Calibrate performance metrics with real data
   - Validate business metric calculations
   - Add confidence intervals
   - Implement A/B testing capability

3. **Model Refinement**:
   - Retrain with historical data
   - Validate feature importance
   - Add model monitoring
   - Implement retraining pipeline

## Features

Current PoC implementation includes:
- 🤖 ML-based profit prediction framework
- 📊 Synthetic data generation for testing
- 🔄 Pluggable data source interface
- 🎯 Hyperparameter tuning system
- 📈 Metric calculation framework
- 📝 Logging and monitoring setup
- 🛠️ CLI for testing and validation

## Mock Performance Metrics

The following metrics are based on synthetic data and serve to validate the system's functionality. These metrics are calculated using simulated product data with realistic distributions but should not be used for actual business decisions.

#### Data Generation Context
- Prices: Generated using beta distribution to mimic real-world price clustering
- Weights: Beta distribution scaled to product-specific ranges
- Reviews: Normal distribution for ratings, negative binomial for counts
- Sales: Product-specific distributions based on market assumptions
- Costs: Calculated using Amazon's FBA fee structure and simulated COGS

#### Training Metrics (from app.log)
- Mean Absolute Error: $25.74 (average dollar difference between predicted and actual profits)
- R² Score: 0.970 (indicates model explains 97% of profit variance)
- Prediction accuracy: 93.2% (percentage of predictions within acceptable error range)
- Predictions within ±10%: 85.9% (percentage of predictions within 10% of actual values)

#### Business Performance Metrics (from summary.json)
- Average Monthly Profit: $274.49 (revenue - costs, including FBA fees and COGS)
- Average ROI: 15.34% (monthly profit / monthly investment in inventory)
- Average Profit Margin: 12.75% (profit as percentage of revenue)
- Break-even Units: 12.36 (units needed to cover fixed and variable costs)
- Average Monthly Sales: 144.83 units (simulated based on market size assumptions)
- Payback Period: 12.35 months (time to recover initial investment)
- Competition Impact: -0.02 (correlation between competitor count and profit)
- Review Impact: 0.03 (correlation between review rating and profit)

#### Summary Statistics
- Total Products Analyzed: 240 (sample size for model validation)
- Average Predicted Profit: $274.56 (mean of model predictions)
- Median Predicted Profit: $246.58 (central tendency, less affected by outliers)
- Profit Range: [-$146.77 to $964.27] (shows potential variance in outcomes)

#### Metric Calculation Methods

**Profit Calculation**
- Monthly Revenue = Units Sold × Price
- Monthly Costs = (COGS + FBA Fees) × Units Sold
- Monthly Profit = Monthly Revenue - Monthly Costs

**ROI and Margins**
- ROI = (Monthly Profit / Monthly Investment) × 100
- Profit Margin = (Monthly Profit / Monthly Revenue) × 100
- Break-even Units = Fixed Costs / (Price - Variable Costs per Unit)

**Impact Correlations**
- Competition Impact: Pearson correlation between competitor count and profit
- Review Impact: Pearson correlation between review rating and profit
- Both metrics range from -1 to 1, with 0 indicating no correlation

#### Interpretation Guidelines

**Model Performance**
- MAE ($25.74) suggests predictions are typically within ~$26 of actual values
- High R² (0.970) indicates strong predictive power, but may be optimistic due to synthetic data
- 85.9% within 10% accuracy suggests good precision for a POC system

**Business Metrics**
- 15.34% ROI represents moderate profitability in simulated conditions
- 12.75% profit margin aligns with typical ecommerce expectations
- 12.36 break-even units suggests reasonable inventory risk
- 12.35 month payback period indicates medium-term investment horizon

**Correlation Insights**
- Weak negative competition impact (-0.02) suggests minimal competitive pressure
- Weak positive review impact (0.03) shows slight benefit from better ratings
- Both correlations are intentionally conservative for POC purposes

#### Limitations and Considerations
1. All metrics are derived from synthetic data with assumed distributions
2. Correlations are artificially constrained to avoid unrealistic relationships
3. Market dynamics are simplified compared to real-world complexity
4. Seasonal effects and trends are not currently modeled
5. External factors (market changes, Amazon policy updates) are not considered

Note: These metrics serve primarily to validate the system's functionality and demonstrate the analytical framework. Real-world implementation will require recalibration with actual historical data.

## Model Architecture and Weights

The system implements a modular model architecture with several key components:

### Base Model Interface
The `Model` abstract base class (`model_interface.py`) defines the core contract:
- Training interface with performance metrics
- Prediction pipeline for new data
- Feature importance calculation
- Model persistence (save/load)
- Business metrics calculation

### Model Implementations

#### 1. Random Forest Model
The primary model (`random_forest_model.py`) includes:

**Feature Weights**
- Price and COGS: Primary drivers of profit calculation
- FBA fees: Direct impact on margins
- Sales volume: Key multiplier for revenue
- Review metrics: Secondary influence
- Competition: Tertiary impact

**Data Processing**
- Feature standardization using StandardScaler
- Categorical encoding for product types
- Missing value handling
- Outlier detection

**Model Parameters**
- Optimized number of trees (50-300 range)
- Dynamic tree depth (3-20 levels)
- Balanced sample splits (2-20 minimum samples)
- Adaptive feature selection (auto/sqrt/log2)
- Optional bootstrapping

#### 2. Gradient Boost Model
Alternative implementation (`gradient_boost_model.py`) using XGBoost:

**Feature Handling**
- Same feature set as Random Forest
- Native categorical feature support
- Gradient-based feature importance
- Efficient memory usage for large datasets

**Model Parameters**
- Large number of trees (100-1000 range)
- Adaptive learning rate (0.01-0.1)
- Leaf-wise tree growth (20-100 leaves)
- Controlled tree depth (3-12 levels)
- Feature and data subsampling

**Key Advantages**
- Faster training times
- Better handling of imbalanced data
- More efficient with high-cardinality features
- Native support for missing values

### Hyperparameter Tuning
The `ModelTuner` class (`model_tuner.py`) implements:

**Optimization Strategy**
- Cross-validation: 5-fold validation by default
- Trial count: 100 optimization attempts
- Objective: Minimize Mean Absolute Error
- Secondary metrics: RMSE and R² score

**Search Spaces**

Random Forest:
```python
{
    'n_estimators': (50, 300),
    'max_depth': (3, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
```

Gradient Boost:
```python
{
    'n_estimators': (100, 1000),
    'learning_rate': (0.01, 0.1),
    'num_leaves': (20, 100),
    'max_depth': (3, 12),
    'min_child_samples': (5, 100),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}
```

**Metric Weights**
- MAE: Primary optimization target
- RMSE: Secondary validation metric
- R²: Model fit indicator
- Business metrics: Post-optimization validation

### Performance Balancing

The system balances multiple objectives:

**Accuracy vs. Speed**
- Default 100 trials for hyperparameter optimization
- Parallel processing for cross-validation
- Model-specific optimizations:
  - Random Forest: Capped tree depth
  - Gradient Boost: Leaf-wise growth

**Precision vs. Generalization**
- Random Forest: Bootstrap sampling
- Gradient Boost: Feature/data subsampling
- Cross-validation for stability
- Feature standardization for consistency

**Business vs. Statistical Metrics**
- MAE optimization for practical error minimization
- ROI and margin validation
- Break-even analysis integration
- Competition and review impact assessment

### Model Selection Criteria

**Random Forest Strengths**:
1. Robust handling of non-linear relationships
2. Built-in feature importance ranking
3. Resistance to overfitting
4. Parallel processing capability
5. Interpretable decision paths

**Gradient Boost Strengths**:
1. Better performance on imbalanced data
2. Faster training and inference
3. Memory-efficient operation
4. Native handling of missing values
5. Fine-grained control over model complexity

Future model implementations can be added by:
1. Implementing the `Model` interface
2. Adding appropriate hyperparameter search spaces
3. Implementing business metric calculations
4. Adding model-specific feature processing

## Model Comparison and Ensemble Framework

The system includes a comprehensive model comparison and ensemble framework for evaluating and combining different prediction approaches.

### Model Comparison

The `ModelEnsemble` class (`model_ensemble.py`) provides automated comparison capabilities:

#### Performance Metrics
- Mean Absolute Error (MAE)
- R² Score (coefficient of determination)
- Root Mean Square Error (RMSE)
- Business metrics (ROI, profit margins, etc.)

#### Visualization Suite
Generated automatically for each comparison:
- MAE comparison bar plots
- R² score visualization
- ROI comparison across models
- Saved as timestamped PNG files

#### Feature Importance Analysis
- Per-model feature rankings
- Importance score comparisons
- Feature stability analysis

### Ensemble Capabilities

#### Weighted Averaging
The system implements a weighted average ensemble:
- Automatic weight optimization based on model performance
- Inverse MAE weighting for better-performing models
- Support for manual weight specification
- Weight persistence and loading

#### Default Models
Two complementary models are included:
1. **Random Forest**:
   - Robust to outliers
   - Handles non-linear relationships
   - Good for feature importance analysis

2. **Gradient Boost (XGBoost)**:
   - Optimized for speed and performance
   - Better compatibility across platforms
   - Built-in early stopping and feature importance
   - Robust to missing values

#### Output Files

The ensemble framework generates several output files:

1. **Comparison Report** (`model_comparison_[timestamp].json`):
```json
{
    "timestamp": "ISO-format timestamp",
    "models": {
        "RandomForestModel": {
            "metrics": {
                "mae": "Mean Absolute Error",
                "r2": "R² Score",
                "average_roi": "ROI percentage",
                ...
            },
            "feature_importance": {
                "feature1": "importance_score",
                ...
            }
        },
        "GradientBoostModel": {
            ...
        }
    },
    "best_model": "Name of best performing model",
    "ensemble_weights": {
        "RandomForestModel": "weight",
        "GradientBoostModel": "weight"
    }
}
```

2. **Visualization File** (`model_comparison_[timestamp].png`):
- Three-panel comparison plot
- MAE, R², and ROI visualizations
- Clear model-to-model comparisons

3. **Ensemble Metadata** (`ensemble_metadata.json`):
```json
{
    "model_paths": ["paths to saved models"],
    "weights": ["model weights"],
    "model_types": ["model class names"]
}
```

### Usage Examples

#### Basic Model Comparison
```python
from src.models.model_ensemble import ModelEnsemble
from src.data.csv_data_source import CSVDataSource

# Load data
data_source = CSVDataSource(
    product_type="envelopes",
    samples=1000,
    seed=42
)
training_data = data_source.get_products()

# Initialize ensemble with default models
ensemble = ModelEnsemble()

# Compare models with hyperparameter tuning
comparison = ensemble.compare_models(
    data=training_data,
    tune_first=True,
    n_trials=100,
    n_folds=5
)

# Access results
best_model = comparison['best_model']
model_weights = comparison['ensemble_weights']

# Analyze feature importance
for model_name, results in comparison['models'].items():
    print(f"\nFeature Importance for {model_name}:")
    for feature, importance in results['feature_importance'].items():
        print(f"{feature}: {importance:.4f}")
```

#### Custom Model Configuration
```python
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boost_model import GradientBoostModel

# Initialize models with custom parameters
rf_model = RandomForestModel(
    model_params={
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5
    }
)

gb_model = GradientBoostModel(
    model_params={
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 50
    }
)

# Create ensemble with custom weights
models = [rf_model, gb_model]
weights = [0.6, 0.4]  # 60% RF, 40% GB
ensemble = ModelEnsemble(models=models, weights=weights)

# Make predictions
predictions = ensemble.predict(new_data)
```

#### Production Workflow Example
```python
import pandas as pd
from pathlib import Path

def analyze_product_profitability(
    data: pd.DataFrame,
    output_dir: str = "predictions",
    tune_models: bool = True
) -> dict:
    """
    End-to-end product profitability analysis workflow.
    
    Args:
        data: Input product data
        output_dir: Directory for saving results
        tune_models: Whether to tune hyperparameters
    
    Returns:
        Dictionary containing analysis results
    """
    # Initialize ensemble
    ensemble = ModelEnsemble(output_dir=output_dir)
    
    # Compare and train models
    comparison = ensemble.compare_models(
        data=data,
        tune_first=tune_models
    )
    
    # Make predictions
    predictions = ensemble.predict(data)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'product_id': data.index,
        'predicted_profit': predictions,
        'confidence': 'medium'  # Placeholder for future enhancement
    })
    
    output_path = Path(output_dir) / "predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    
    # Save ensemble for future use
    ensemble.save(output_dir)
    
    return {
        'comparison': comparison,
        'predictions_path': str(output_path),
        'best_model': comparison['best_model'],
        'model_weights': comparison['ensemble_weights']
    }

# Usage
results = analyze_product_profitability(
    data=training_data,
    output_dir="production/models",
    tune_models=True
)
```

### Best Practices

#### Model Selection and Tuning
1. **Data Volume Considerations**
   - Use Random Forest for smaller datasets (<10K samples)
   - Prefer Gradient Boost for larger datasets (>10K samples)
   - Adjust `n_trials` based on available computation time

2. **Feature Engineering**
   - Normalize price and weight features
   - One-hot encode categorical variables
   - Handle missing values consistently
   - Consider feature interactions for complex relationships

3. **Model Weights**
   - Start with automated weight optimization
   - Adjust weights based on:
     * Historical model performance
     * Data quality per feature
     * Business requirements
     * Prediction confidence

4. **Performance Monitoring**
   - Track key metrics over time:
     * MAE trend
     * R² stability
     * Feature importance shifts
     * Business metric alignment
   - Monitor for model drift
   - Validate predictions against actuals

5. **Business Integration**
   - Align model selection with business goals:
     * Prioritize MAE for direct profit prediction
     * Focus on R² for trend analysis
     * Consider ROI for investment decisions
   - Validate predictions against:
     * Historical performance
     * Market conditions
     * Seasonal patterns
     * Competition metrics

6. **Validation Strategy**
   - Use cross-validation for stable metrics
   - Implement time-based validation when possible
   - Test extreme cases and edge scenarios
   - Validate business metric calculations

7. **Error Handling**
   - Log all prediction failures
   - Track feature distribution changes
   - Monitor for outlier predictions
   - Implement fallback strategies:
     * Use ensemble average on model failure
     * Maintain backup model versions
     * Set reasonable prediction bounds

8. **Documentation and Reproducibility**
   - Record all hyperparameter choices
   - Document feature transformations
   - Save model comparison results
   - Track data preprocessing steps
   - Maintain version control for:
     * Model implementations
     * Training datasets
     * Prediction outputs
     * Configuration files

9. **Deployment Considerations**
   - Version models appropriately
   - Implement gradual rollout
   - Monitor system resources
   - Plan for model updates
   - Consider:
     * Prediction latency
     * Memory usage
     * Batch vs. real-time
     * Backup procedures

10. **Maintenance and Updates**
    - Schedule regular retraining
    - Monitor prediction quality
    - Update feature distributions
    - Maintain test coverage
    - Review and update:
      * Business metrics
      * Model weights
      * Validation thresholds
      * Performance targets

## Validation Framework

The system employs a multi-layered validation approach:

### Statistical Validation
- 80/20 train-validation split for model evaluation
- Feature standardization using StandardScaler
- Cross-validation during hyperparameter tuning
- Multiple performance metrics (MAE, RMSE, R²)

### Business Logic Validation
- ROI and margin calculations for each prediction
- Break-even analysis validation
- Payback period calculations
- Competition impact correlation
- Review-to-profit relationship validation

### Data Quality Validation
- Schema validation for input data
- Feature range and type checking
- Missing value detection
- Categorical value validation

### Monitoring and Logging
- Detailed metric logging in app.log
- Training/validation split performance tracking
- Feature importance monitoring
- Error and warning tracking

### Business Metrics Framework

The analyzer implements calculations for:

#### Financial Metrics
- ROI calculation framework
- Margin analysis system
- Break-even analysis
- Net profit calculation

#### Operational Metrics
- Sales volume tracking
- Payback period estimation
- Competition impact analysis
- Ranking correlation framework
- Review impact analysis

## Development

### Current Stack
- scikit-learn for ML pipeline
- pandas for data handling
- Click for CLI
- loguru for logging
- numpy for numerical operations

### Adding Real Data Sources

To add real data sources:
1. Implement `DataSource` interface in `src/data/data_source.py`
2. Add data validation in `src/data/validators/`
3. Update feature preprocessing in `src/models/`
4. Modify metric calculations as needed

## Project Structure

```
ecomm_oracle/
├── src/
│   ├── data/
│   │   ├── data_source.py       # Abstract data source interface
│   │   ├── mock_data_source.py  # Synthetic data generation (to be replaced)
│   │   ├── csv_data_source.py   # CSV data source implementation
│   │   ├── real_data_source.py  # Real data source template
│   │   └── product_specs.py     # Product specifications interface
│   ├── models/
│   │   ├── model_interface.py   # Base model interface
│   │   ├── model_tuner.py       # Hyperparameter tuning framework
│   │   ├── random_forest_model.py # Random Forest implementation
│   │   ├── gradient_boost_model.py # XGBoost implementation
│   │   └── model_ensemble.py    # Model comparison and ensemble framework
│   └── utils/
│       └── logger.py            # Logging configuration
├── data/                        # Data storage directory
├── models/                      # Saved model directory
├── predictions/                 # Prediction outputs
│   ├── *_predictions.csv       # Detailed predictions
│   └── *_summary.json         # Summary metrics
└── logs/                       # Log files
    ├── app.log                # Application logs
    └── error.log             # Error tracking
```

### Key Components

#### Model Layer
- `model_interface.py`: Abstract base class defining model contract
- `random_forest_model.py`: Primary model implementation
- `gradient_boost_model.py`: Alternative model implementation
- `model_ensemble.py`: Model comparison and weighted ensemble
- `model_tuner.py`: Hyperparameter optimization framework

#### Data Layer
- `data_source.py`: Abstract interface for data sources
- `mock_data_source.py`: Synthetic data generation (POC)
- `csv_data_source.py`: CSV-based data source
- `real_data_source.py`: Template for production data source
- `product_specs.py`: Product type specifications

#### Utility Layer
- `logger.py`: Centralized logging configuration
- CLI interface in `main.py`
- Error handling and monitoring

### Directory Structure

#### Source Code (`src/`)
Contains all Python modules organized by functionality:
- `data/`: Data handling and generation
- `models/`: ML models and ensemble framework
- `utils/`: Shared utilities

#### Data Storage (`data/`)
- Input data storage
- Generated synthetic data
- Data source files

#### Model Storage (`models/`)
- Saved model states
- Tuning results
- Ensemble configurations

#### Predictions (`predictions/`)
- Detailed prediction CSVs
- Summary metrics JSON
- Model comparison results
- Visualization outputs

#### Logs (`logs/`)
- Application logs
- Error tracking
- Performance metrics
- Execution timing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ecomm_oracle
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The tool provides a command-line interface with two main commands:

### List Available Product Types

```bash
python main.py list-products
```

### Analyze Product Profitability

Basic usage with default settings:
```bash
python main.py analyze <product_type>
```

Example with specific product type:
```bash
python main.py analyze envelopes
```

### Advanced Options

- Generate more samples:
```bash
python main.py analyze envelopes --samples 2000
```

- Enable hyperparameter tuning:
```bash
python main.py analyze envelopes --tune
```

- Use existing model for predictions only:
```bash
python main.py analyze envelopes --predict-only
```

- Customize directories:
```bash
python main.py analyze envelopes --model-dir custom_models --data-dir custom_data --output-dir custom_predictions
```

### Command Line Options

```
Options:
  -n, --samples INTEGER     Number of samples to generate [default: 1000]
  --model-dir TEXT         Directory for model files [default: models]
  --data-dir TEXT         Directory for data files [default: data]
  --output-dir TEXT       Directory for prediction outputs [default: predictions]
  --tune / --no-tune      Tune model hyperparameters [default: no-tune]
  --predict-only          Only make predictions using existing model
  --help                  Show this message and exit
```

## Data Generation

The system generates realistic product data using:
- Beta distributions for prices and weights
- Normal distributions for review ratings
- Negative binomial distributions for review counts
- Competition-based distributions for competitor counts
- Product-specific parameters for FBA fees and margins

## Output Files

The analyzer generates three types of output files:

1. **Log File** (`logs/app.log`):
   - Training metrics (MAE, R², RMSE)
   - Business performance metrics
   - Prediction accuracy percentages
   - Data validation results
   - Execution timing information
   - Error and warning messages

2. **Detailed Predictions** (`predictions/<product_type>_predictions.csv`):
   - Product identifiers (ASIN)
   - Category and subcategory information
   - Product features (price, weight, review metrics)
   - Market data (competitors, estimated sales)
   - Cost breakdown (FBA fees, COGS)
   - Monthly profit predictions
   - Last updated timestamp

3. **Summary Report** (`predictions/<product_type>_summary.json`):
   - Product type identifier
   - Total number of products analyzed
   - Average and median predicted profits
   - Profit range [min, max]
   - Business performance metrics
   - Model performance indicators

Additionally, errors and exceptions are logged to `logs/error.log` for debugging purposes.
