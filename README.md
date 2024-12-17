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

The following metrics are based on synthetic data and serve to validate the system's functionality:

#### System Validation
- 85.9% of predictions within ±10% (synthetic baseline)
- Mean Absolute Error: $25.74
- R² Score: 0.970 (on synthetic data)
- Overall prediction accuracy: 93.2%

#### Sample Profit Analysis
- Test profit range: $-221.26 to $1,187.10
- Sample mean profit: $260.76
- Sample median profit: $246.58
- Total products analyzed: 240

Note: These metrics are generated from synthetic data and should not be used for actual business decisions.

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
│   └── utils/
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

## Output

The analyzer generates two types of output files for each analysis:

1. **Detailed Predictions** (`predictions/<product_type>_predictions.csv`):
   - Product identifiers (ASIN)
   - Category and subcategory information
   - Product features (price, weight, review metrics)
   - Market data (competitors, estimated sales)
   - Cost breakdown (FBA fees, COGS)
   - Monthly profit predictions
   - Last updated timestamp

2. **Summary Report** (`predictions/<product_type>_summary.json`):
   - Product type identifier
   - Total number of products analyzed
   - Average predicted profit
   - Median predicted profit
   - Profit range [min, max]

The system also maintains detailed logs in `logs/app.log` with:
   - Training and prediction metrics
   - Data validation results
   - Execution times
   - Error tracking and debugging information
