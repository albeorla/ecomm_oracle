"""Data generation utilities for synthetic Amazon product data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Define constants for data generation
CATEGORIES = [
    'Electronics',
    'Home & Kitchen',
    'Toys & Games',
    'Sports & Outdoors',
    'Office & Supplies'
]

# More realistic price ranges based on category
PRICE_RANGES = {
    'Electronics': (15.99, 199.99),
    'Home & Kitchen': (9.99, 79.99),
    'Toys & Games': (7.99, 49.99),
    'Sports & Outdoors': (12.99, 89.99),
    'Office & Supplies': (8.99, 24.99)
}

# More realistic weight ranges (in pounds)
WEIGHT_RANGES = {
    'Electronics': (0.2, 3),
    'Home & Kitchen': (0.3, 5),
    'Toys & Games': (0.1, 2),
    'Sports & Outdoors': (0.2, 4),
    'Office & Supplies': (0.2, 0.8)
}

# Seasonal factors by category (Q1, Q2, Q3, Q4)
SEASONAL_FACTORS = {
    'Electronics': [0.8, 0.7, 0.9, 1.6],  # Holiday heavy
    'Home & Kitchen': [1.0, 1.1, 0.9, 1.0],  # Fairly stable
    'Toys & Games': [0.6, 0.7, 0.8, 1.9],  # Very holiday heavy
    'Sports & Outdoors': [0.9, 1.4, 1.2, 0.5],  # Summer heavy
    'Office & Supplies': [1.3, 0.9, 1.4, 0.8]
}

def generate_sample_data(n_samples=1000, save_to_file=False):
    """Generate synthetic Amazon product data for POC.
    
    Args:
        n_samples (int): Number of samples to generate
        save_to_file (bool): If True, saves data to data/generated directory
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    np.random.seed(42)
    
    # Generate categories first to use for other features
    categories = np.random.choice(CATEGORIES, n_samples)
    
    # Generate basic product features
    data = {
        'asin': [f'B{str(i).zfill(9)}' for i in range(n_samples)],
        'category': categories,
        'price': np.array([
            np.random.uniform(*PRICE_RANGES[cat]) for cat in categories
        ]),
        'weight': np.array([
            np.random.uniform(*WEIGHT_RANGES[cat]) for cat in categories
        ]),
        'review_rating': np.random.normal(4.2, 0.4, n_samples).clip(1, 5),  # Most products 3.8-4.6 stars
        'review_count': np.random.negative_binomial(5, 0.3, n_samples),  # Realistic review counts
        'competitors': np.random.randint(3, 15, n_samples),  # Fewer competitors for office supplies
        'estimated_monthly_sales': np.random.negative_binomial(4, 0.2, n_samples) + 10,  # Higher base sales
        'seller_count': np.random.negative_binomial(2, 0.3, n_samples) + 1,  # Number of sellers
        'bsr': np.random.negative_binomial(8, 0.1, n_samples) * 1000,  # Best Sellers Rank
        'review_velocity': np.random.negative_binomial(2, 0.4, n_samples),  # New reviews per month
    }
    
    # Add seasonal factors
    current_quarter = (datetime.now().month - 1) // 3
    data['seasonal_factor'] = [SEASONAL_FACTORS[cat][current_quarter] for cat in categories]
    
    # Calculate FBA fees (based on Amazon's fee structure)
    # Base fee + weight-based fee + category-specific fee
    base_fee = 2.92  # Lower base fee for small items
    weight_fee = data['weight'] * 0.40  # Lower per-pound fee for envelopes
    category_fees = {
        'Electronics': 1.0,
        'Home & Kitchen': 0.8,
        'Toys & Games': 0.6,
        'Sports & Outdoors': 0.9,
        'Office & Supplies': 0.4
    }
    data['fba_fees'] = base_fee + weight_fee + np.array([category_fees[cat] for cat in categories])
    
    # Calculate Amazon referral fees (15% for most categories)
    referral_fees = data['price'] * 0.15
    
    # Estimate PPC costs (8-15% for office supplies - less competitive)
    ppc_costs = np.where(
        np.array(categories) == 'Office & Supplies',
        data['price'] * np.random.uniform(0.08, 0.15, n_samples),
        data['price'] * np.random.uniform(0.10, 0.20, n_samples)
    )
    
    # Generate COGS (25-40% for office supplies - better margins)
    data['cogs'] = np.where(
        np.array(categories) == 'Office & Supplies',
        data['price'] * np.random.uniform(0.25, 0.40, n_samples),
        data['price'] * np.random.uniform(0.30, 0.50, n_samples)
    )
    
    # Apply seasonal factor to sales
    data['estimated_monthly_sales'] = data['estimated_monthly_sales'] * data['seasonal_factor']
    
    # Calculate profitability (target variable)
    data['monthly_profit'] = (
        (data['price'] - data['cogs'] - data['fba_fees'] - referral_fees - ppc_costs) * 
        data['estimated_monthly_sales']
    )
    
    df = pd.DataFrame(data)
    
    if save_to_file:
        # Save to data/generated directory
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'generated'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'synthetic_data_{timestamp}.csv'
        df.to_csv(output_path, index=False)
    
    return df