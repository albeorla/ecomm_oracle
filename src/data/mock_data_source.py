from typing import Dict, Any
import numpy as np
import pandas as pd
from .data_source import DataSource
from .product_generator import ProductGenerator

class MockDataSource(DataSource):
    """Mock data source using synthetic data generation."""
    
    def __init__(self, product_type: str, n_samples: int = 1000, validation_split: float = 0.2):
        """
        Initialize mock data source.
        
        Args:
            product_type: Type of product to generate
            n_samples: Number of samples to generate
            validation_split: Fraction of data to use for validation
        """
        self.generator = ProductGenerator()
        self.product_type = product_type
        self.n_samples = n_samples
        self.validation_split = validation_split
        
    def get_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data."""
        full_data = self.generator.generate_products(
            int(self.n_samples * (1 + self.validation_split)), 
            self.product_type
        )
        train_size = int(len(full_data) * (1 - self.validation_split))
        return full_data.iloc[:train_size]
    
    def get_validation_data(self) -> pd.DataFrame:
        """Generate synthetic validation data."""
        full_data = self.generator.generate_products(
            int(self.n_samples * (1 + self.validation_split)), 
            self.product_type
        )
        train_size = int(len(full_data) * (1 - self.validation_split))
        return full_data.iloc[train_size:]
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate data schema matches requirements."""
        required_columns = {
            'price': np.float64,
            'weight': np.float64,
            'review_rating': np.float64,
            'review_count': np.int64,
            'competitors': np.int64,
            'estimated_monthly_sales': np.int64,
            'fba_fees': np.float64,
            'cogs': np.float64,
            'monthly_profit': np.float64
        }
        
        for col, dtype in required_columns.items():
            if col not in data.columns:
                return False
            if not np.issubdtype(data[col].dtype, dtype):
                return False
        
        return True
    
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
            }
        } 