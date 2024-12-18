from typing import Dict, Any
import numpy as np
import pandas as pd
from .data_source import DataSource
from .data_generator import generate_sample_data

class MockDataSource(DataSource):
    """Mock data source using synthetic data generation."""
    
    # Map specific product types to categories
    PRODUCT_TYPE_MAP = {
        'electronics': 'Electronics',
        'kitchen': 'Home & Kitchen',
        'toys': 'Toys & Games',
        'sports': 'Sports & Outdoors',
        'envelopes': 'Office & Supplies',
        'paper': 'Office & Supplies',
        'office': 'Office & Supplies'
    }
    
    def __init__(self, product_type: str, n_samples: int = 1000):
        """Initialize the mock data source.
        
        Args:
            product_type (str): Type of product to generate data for
            n_samples (int): Number of samples to generate
        """
        self.product_type = product_type.lower()
        self.n_samples = n_samples
        self.category = self.PRODUCT_TYPE_MAP.get(self.product_type)
        if not self.category:
            raise ValueError(f"Unknown product type: {product_type}")
        
        # Generate full dataset
        self.data = self._get_data()
    
    def _get_data(self):
        """Generate synthetic data for the specified product type."""
        # Generate base data
        df = generate_sample_data(self.n_samples)
        
        # Filter to only include rows from our category
        df = df[df['category'] == self.category].copy()
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def get_training_data(self):
        """Get training data subset."""
        n_train = int(len(self.data) * 0.8)
        return self.data.head(n_train)
    
    def get_validation_data(self):
        """Get validation data subset."""
        n_train = int(len(self.data) * 0.8)
        return self.data.tail(len(self.data) - n_train)
    
    def validate_schema(self, df):
        """Validate the DataFrame schema.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if schema is valid
        """
        required_columns = {
            'asin': (str, np.str_, object),
            'category': (str, np.str_, object),
            'price': (int, float, np.int32, np.int64, np.float32, np.float64),
            'weight': (int, float, np.int32, np.int64, np.float32, np.float64),
            'review_rating': (int, float, np.int32, np.int64, np.float32, np.float64),
            'review_count': (int, float, np.int32, np.int64, np.float32, np.float64),
            'competitors': (int, float, np.int32, np.int64, np.float32, np.float64),
            'estimated_monthly_sales': (int, float, np.int32, np.int64, np.float32, np.float64),
            'fba_fees': (int, float, np.int32, np.int64, np.float32, np.float64),
            'cogs': (int, float, np.int32, np.int64, np.float32, np.float64),
            'monthly_profit': (int, float, np.int32, np.int64, np.float32, np.float64),
            'seller_count': (int, float, np.int32, np.int64, np.float32, np.float64),
            'bsr': (int, float, np.int32, np.int64, np.float32, np.float64),
            'review_velocity': (int, float, np.int32, np.int64, np.float32, np.float64),
            'seasonal_factor': (int, float, np.int32, np.int64, np.float32, np.float64)
        }
        
        # Check all required columns exist
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Missing columns: {missing_cols}")
            return False
        
        # Check column types
        for col, expected_types in required_columns.items():
            if not df[col].dtype in [np.dtype(t) for t in expected_types] and not isinstance(df[col].iloc[0], expected_types):
                print(f"Column {col} has wrong type: {df[col].dtype}")
                return False
        
        return True
    
    def validate_features(self, df):
        """Validate feature values.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if features are valid
        """
        try:
            # Check for nulls
            if df.isnull().any().any():
                return False
            
            # Check value ranges
            validations = [
                (df['price'] > 0).all(),
                (df['weight'] > 0).all(),
                (df['review_rating'] >= 1).all() and (df['review_rating'] <= 5).all(),
                (df['review_count'] >= 0).all(),
                (df['competitors'] >= 0).all(),
                (df['estimated_monthly_sales'] >= 0).all(),
                (df['fba_fees'] > 0).all(),
                (df['cogs'] > 0).all(),
                (df['seller_count'] > 0).all(),
                (df['bsr'] > 0).all(),
                (df['review_velocity'] >= 0).all(),
                (df['seasonal_factor'] > 0).all()
            ]
            
            return all(validations)
            
        except Exception:
            return False
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get feature specifications."""
        return {
            'price': {
                'type': np.float64,
                'min': 0.0,
                'description': 'Product selling price'
            },
            'weight': {
                'type': np.float64,
                'min': 0.0,
                'description': 'Product weight in pounds'
            },
            'review_rating': {
                'type': np.float64,
                'min': 1.0,
                'max': 5.0,
                'description': 'Average product rating'
            },
            'review_count': {
                'type': np.int64,
                'min': 0,
                'description': 'Number of product reviews'
            },
            'competitors': {
                'type': np.int64,
                'min': 0,
                'description': 'Number of competing products'
            },
            'estimated_monthly_sales': {
                'type': np.int64,
                'min': 0,
                'description': 'Estimated monthly sales volume'
            },
            'fba_fees': {
                'type': np.float64,
                'min': 0.0,
                'description': 'Amazon FBA fees'
            },
            'cogs': {
                'type': np.float64,
                'min': 0.0,
                'description': 'Cost of goods sold'
            },
            'monthly_profit': {
                'type': np.float64,
                'description': 'Monthly profit'
            },
            'category': {
                'type': object,
                'description': 'Product category'
            }
        } 