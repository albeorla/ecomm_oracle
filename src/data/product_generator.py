import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta
from .product_specs import ProductSpecs, ProductTypeSpec
from loguru import logger

class ProductGenerator:
    """Generate realistic Amazon product data for specific product types."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the product generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def _generate_prices(self, n_samples: int, price_range: Tuple[float, float]) -> np.ndarray:
        """Generate product prices using beta distribution."""
        price_min, price_max = price_range
        return np.random.beta(2, 5, n_samples) * (price_max - price_min) + price_min
    
    def _generate_weights(self, n_samples: int, weight_range: Tuple[float, float]) -> np.ndarray:
        """Generate product weights using beta distribution."""
        weight_min, weight_max = weight_range
        return np.random.beta(2, 5, n_samples) * (weight_max - weight_min) + weight_min
    
    def _generate_review_ratings(self, n_samples: int, rating_mean: float) -> np.ndarray:
        """Generate review ratings using normal distribution."""
        return np.clip(np.random.normal(rating_mean, 0.3, n_samples), 1, 5)
    
    def _generate_review_counts(self, n_samples: int, count_mean: float) -> np.ndarray:
        """Generate review counts using negative binomial distribution."""
        return np.random.negative_binomial(
            n=5,
            p=count_mean / (count_mean + 5),
            size=n_samples
        )
    
    def _generate_competitors(self, n_samples: int, competition_level: str) -> np.ndarray:
        """Generate competitor counts based on competition level."""
        comp_params = ProductSpecs.COMPETITION_LEVELS[competition_level]
        return np.clip(
            np.random.normal(comp_params['mean'], comp_params['std'], n_samples),
            1, None
        ).astype(int)
    
    def _generate_sales(self, n_samples: int, sales_velocity: str) -> np.ndarray:
        """Generate monthly sales based on sales velocity."""
        sales_params = ProductSpecs.SALES_VELOCITY[sales_velocity]
        return np.clip(
            np.random.normal(sales_params['mean'], sales_params['std'], n_samples),
            0, None
        ).astype(int)
    
    def _calculate_fba_fees(self, weights: np.ndarray, prices: np.ndarray, fee_multiplier: float) -> np.ndarray:
        """Calculate FBA fees based on weight and price."""
        base_fee = 3.0
        weight_factor = weights * 2.5
        return (base_fee + weight_factor) * fee_multiplier
    
    def generate_products(self, n_samples: int, product_type: str) -> pd.DataFrame:
        """
        Generate synthetic product data.
        
        Args:
            n_samples: Number of products to generate
            product_type: Type of product to generate
            
        Returns:
            DataFrame containing generated product data
        """
        logger.info(f"Generating {n_samples} {product_type} products")
        
        # Get product specifications
        spec = ProductSpecs.get_spec(product_type)
        
        # Generate base features
        df = pd.DataFrame({
            'asin': [f"B{str(i).zfill(9)}" for i in range(n_samples)],
            'category': spec.category,
            'subcategory': spec.subcategory,
            'price': self._generate_prices(n_samples, spec.price_range),
            'weight': self._generate_weights(n_samples, spec.weight_range),
            'review_rating': self._generate_review_ratings(n_samples, spec.review_rating_mean),
            'review_count': self._generate_review_counts(n_samples, spec.review_count_mean),
            'competitors': self._generate_competitors(n_samples, spec.competition_level),
            'estimated_monthly_sales': self._generate_sales(n_samples, spec.sales_velocity),
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        })
        
        # Calculate derived features
        df['fba_fees'] = self._calculate_fba_fees(df['weight'], df['price'], spec.fba_fee_multiplier)
        df['cogs'] = df['price'] * (1 - spec.typical_margin)
        df['monthly_profit'] = (
            (df['price'] - df['fba_fees'] - df['cogs']) * 
            df['estimated_monthly_sales']
        )
        
        logger.debug(f"Generated {len(df)} products with average profit: ${df['monthly_profit'].mean():.2f}")
        return df 