# Amazon FBA Product Profitability Analyzer

A machine learning tool to analyze and predict profitability of Amazon FBA products. Helps identify the most promising ASINs within a category based on profitability, competition, and market metrics.

## Key Features

1. **Category Viability Analysis**:
   - Percentage of profitable products
   - Number of high-ROI opportunities
   - Competition levels
   - Average and median profits
   - Risk assessment (profit volatility)

2. **ASIN Scoring System (0-15 points)**:
   - Profitability (0-5 points)
     - Monthly profit thresholds
     - ROI targets (40%, 60%)
   - Competition (0-3 points)
     - Number of competitors
     - Seller count
   - Demand (0-4 points)
     - BSR thresholds
     - Monthly sales volume
   - Reviews (0-3 points)
     - Rating threshold
     - Review count
     - Review velocity

3. **Cost Structure Analysis**:
   - FBA fees calculation
   - PPC cost estimation
   - COGS modeling
   - Profit margin analysis

## Example Workflow

1. **Analyze a Product Category**:
```bash
python main.py analyze envelopes
```

Expected Output:
```
Model Training Metrics:
Mean Absolute Error: $48.24
R² Score: 0.974
RMSE: $78.97

Category Viability Summary:
Total Products Analyzed: 41
Profitable Products: 65.2%
Products with High ROI: 12
Products with Low Competition: 15
Products with Good BSR: 18
Average Monthly Profit: $127.45
Top 10 Average Profit: $312.67

Top 5 Opportunities:
ASIN: B000000528 | Score: 13/15 | Profit: $74.57 | ROI: 52.0% | BSR: 34,000 | Competitors: 6
ASIN: B000000003 | Score: 12/15 | Profit: $69.12 | ROI: 48.5% | BSR: 41,000 | Competitors: 8
...
```

2. **Output Files**:
- `predictions/envelopes_predictions.csv`: Detailed ASIN-level data
  ```
  asin,category,price,monthly_profit,roi,asin_score,bsr,competitors,...
  B000000528,Office & Supplies,24.99,74.57,0.52,13,34000,6,...
  ```
- `predictions/envelopes_summary.json`: Category metrics
  ```json
  {
    "product_type": "envelopes",
    "total_products": 41,
    "profitable_percentage": "65.2%",
    "high_roi_count": 12,
    "avg_monthly_profit": 127.45,
    ...
  }
  ```

## Success Criteria

1. **Category Level**:
   - \>50% profitable products
   - \>10 high-ROI opportunities
   - Average profit >$100/month
   - Low profit volatility

2. **Individual ASINs**:
   - ASIN Score >10/15
   - ROI >40%
   - BSR <50,000
   - <10 competitors
   - >10 monthly sales

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/albeorla/ecomm_oracle.git
cd ecomm_oracle
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. List Available Product Types
```bash
python main.py list-products
```

5. Analyze a Product Type
```bash
python main.py analyze [PRODUCT_TYPE] [OPTIONS]
```

Options:
- `-n, --samples`: Number of samples to generate (default: 1000)
- `--tune/--no-tune`: Enable/disable hyperparameter tuning
- `--predict-only`: Use existing model without retraining

Example:
```bash
python main.py analyze electronics -n 2000 --tune
```

## Project Structure

```
ecomm_oracle/
├── data/               # Data storage and generation
│   ├── generated/     # Generated synthetic data
│   └── product_specs/ # Product category specifications
├── models/            # Saved model files
├── output/           # Analysis outputs
│   ├── logs/        # Application logs
│   └── predictions/ # Prediction results
├── src/             # Source code
│   ├── data/        # Data handling modules
│   └── models/      # ML model implementations
├── main.py          # CLI application
└── requirements.txt # Project dependencies
```

## Business Metrics

The analyzer provides realistic business metrics based on current FBA market conditions:

1. **Category-Specific Metrics**

   Office & Supplies:
   - Price Range: $8.99-$24.99
   - Target ROI: 40-80%
   - Profit Margins: 20-35%
   - Monthly Sales: 10-50 units
   - Seasonal Peaks: Q1 (tax season), Q3 (back-to-school)
   - PPC Costs: 8-15% of revenue
   - COGS: 25-40% of price

   Electronics:
   - Price Range: $15.99-$199.99
   - Target ROI: 30-60%
   - Profit Margins: 15-25%
   - Seasonal Peak: Q4 (holidays)
   - PPC Costs: 10-20% of revenue
   - COGS: 30-50% of price

2. **Cost Structure**
   - Amazon Referral Fees: 15% of price
   - FBA Fees: Based on size/weight
     - Base fee: $2.92-$3.00
     - Weight fee: $0.40-$0.75/lb
     - Category fee: $0.4-$1.0
   - PPC Costs: Category dependent
   - COGS: Category dependent

3. **Market Impact**
   - Competition Impact: Correlation with profit
   - Review Rating Impact: Correlation with profit
   - Payback Period: 1.5-3 months typical

4. **Success Indicators**
   - BSR < 50,000 in category
   - Review rating > 4.0
   - Monthly sales > 10 units
   - Profit margin > 20%
   - ROI > 40%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
