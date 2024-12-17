import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate synthetic Amazon product data for POC."""
    np.random.seed(42)
    
    # Generate basic product features
    data = {
        'asin': [f'B{str(i).zfill(9)}' for i in range(n_samples)],
        'category': np.random.choice(['Electronics', 'Home & Kitchen', 'Toys & Games', 'Sports & Outdoors'], n_samples),
        'price': np.random.uniform(10, 200, n_samples),
        'weight': np.random.uniform(0.1, 10, n_samples),
        'review_rating': np.random.uniform(1, 5, n_samples),
        'review_count': np.random.randint(0, 1000, n_samples),
        'competitors': np.random.randint(1, 20, n_samples),
        'estimated_monthly_sales': np.random.randint(0, 500, n_samples),
    }
    
    # Calculate FBA fees (simplified)
    data['fba_fees'] = data['weight'] * 2.5 + 3.0
    
    # Generate COGS (Cost of Goods Sold)
    data['cogs'] = data['price'] * np.random.uniform(0.2, 0.4, n_samples)
    
    # Calculate profitability (target variable)
    data['monthly_profit'] = (
        (data['price'] - data['cogs'] - data['fba_fees']) * data['estimated_monthly_sales']
    )
    
    return pd.DataFrame(data) 