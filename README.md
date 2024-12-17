# Amazon FBA Product Profitability Analyzer

A machine learning-powered tool for predicting and analyzing profitability of Amazon FBA (Fulfillment by Amazon) products. This project serves as a proof-of-concept for data-driven decision making in e-commerce product selection.

## Features

- 🤖 Machine learning-based profit prediction (93% accuracy)
- 📊 Statistical analysis of product performance
- 🔄 Automated data generation with realistic distributions
- 🎯 Hyperparameter tuning for optimal model performance
- 📈 Detailed performance metrics and predictions
- 📝 Comprehensive logging and execution timing
- 🛠️ Command-line interface with sensible defaults

## Performance Metrics

Based on recent testing with envelope products:
- 86.4% of predictions within 10% accuracy
- Mean Absolute Error: $11.88
- Overall prediction accuracy: 93.17%
- Profit range: -$220 to $1,184
- Average predicted profit: $260.73

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

## Project Structure

```
ecomm_oracle/
├── src/
│   ├── data/
│   │   ├── csv_data_source.py    # Data generation and handling
│   │   ├── product_generator.py  # Realistic data generation
│   │   └── product_specs.py      # Product specifications
│   ├── models/
│   │   ├── model_interface.py    # Model interface definition
│   │   ├── model_tuner.py       # Hyperparameter tuning
│   │   └── random_forest_model.py # ML model implementation
│   └── utils/
│       └── logger.py             # Logging configuration
├── data/                         # Generated data files
├── models/                       # Trained model files
├── predictions/                  # Prediction outputs
├── main.py                      # CLI entry point
└── requirements.txt             # Project dependencies
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
   - Individual predictions for each product
   - Actual vs predicted profits
   - Prediction accuracy metrics
   - Feature values used for prediction

2. **Summary Report** (`predictions/<product_type>_summary.json`):
   - Overall statistics and metrics
   - Average and median predicted profit
   - Profit range and distribution
   - Model performance indicators
   - Percentage of predictions within accuracy thresholds

## Development

The project uses:
- scikit-learn for machine learning
- pandas for data handling
- Click for CLI interface
- loguru for logging
- numpy for numerical operations
