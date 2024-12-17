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
│   │   └── product_specs.py     # Product specifications interface
│   ├── models/
│   │   ├── model_interface.py   # Model interface definition
│   │   ├── model_tuner.py      # Tuning framework
│   │   └── random_forest_model.py
│   └─ utils/
│       └── logger.py
```

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
