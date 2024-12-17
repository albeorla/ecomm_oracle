from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProductTypeSpec:
    """Specification for a product type."""
    name: str
    category: str
    subcategory: str
    price_range: tuple[float, float]
    weight_range: tuple[float, float]
    typical_margin: float  # As decimal, e.g., 0.4 for 40%
    fba_fee_multiplier: float  # Relative to base FBA fee
    seasonality: List[float]  # 12 monthly factors, 1.0 is baseline
    competition_level: str  # 'low', 'medium', 'high'
    review_rating_mean: float
    review_count_mean: float
    sales_velocity: str  # 'slow', 'medium', 'fast'

class ProductSpecs:
    """Product specifications for different product types."""
    
    # Competition level definitions
    COMPETITION_LEVELS = {
        'low': {'mean': 3, 'std': 1},
        'medium': {'mean': 8, 'std': 2},
        'high': {'mean': 15, 'std': 4}
    }
    
    # Sales velocity definitions (monthly sales)
    SALES_VELOCITY = {
        'slow': {'mean': 50, 'std': 20},
        'medium': {'mean': 150, 'std': 50},
        'fast': {'mean': 400, 'std': 100}
    }
    
    # Product type specifications
    SPECS = {
        'envelopes': ProductTypeSpec(
            name='Envelopes',
            category='Office Products',
            subcategory='Envelopes & Shipping Supplies',
            price_range=(8.99, 24.99),
            weight_range=(0.2, 2.0),  # in pounds
            typical_margin=0.45,  # 45% margin
            fba_fee_multiplier=0.9,  # Lower than average due to easy handling
            seasonality=[  # Monthly factors (Jan-Dec)
                1.2,  # Jan: Tax season prep
                1.3,  # Feb: Tax season peak
                1.1,  # Mar: Tax season trailing
                1.0,  # Apr
                0.9,  # May
                0.8,  # Jun
                0.8,  # Jul
                1.1,  # Aug: Back to school
                1.0,  # Sep
                0.9,  # Oct
                1.0,  # Nov
                1.1   # Dec: Holiday cards
            ],
            competition_level='medium',
            review_rating_mean=4.3,
            review_count_mean=200,
            sales_velocity='medium'
        ),
        # Add more product types here following the same pattern
        # Example:
        # 'phone_cases': ProductTypeSpec(...)
    }
    
    @classmethod
    def get_spec(cls, product_type: str) -> ProductTypeSpec:
        """Get specification for a product type."""
        if not cls.is_valid_product_type(product_type):
            raise ValueError(f"Unknown product type: {product_type}")
        return cls.SPECS[product_type]
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available product types."""
        return list(cls.SPECS.keys())
    
    @classmethod
    def is_valid_product_type(cls, product_type: str) -> bool:
        """Check if a product type is valid."""
        return product_type in cls.SPECS
    
    @classmethod
    def add_product_type(cls, name: str, spec: ProductTypeSpec) -> None:
        """
        Add a new product type specification.
        
        Args:
            name: Name/key for the product type
            spec: ProductTypeSpec instance with the specifications
        """
        if not isinstance(spec, ProductTypeSpec):
            raise TypeError("spec must be a ProductTypeSpec instance")
        
        if len(spec.seasonality) != 12:
            raise ValueError("seasonality must have exactly 12 monthly factors")
        
        if spec.competition_level not in cls.COMPETITION_LEVELS:
            raise ValueError(f"Invalid competition level: {spec.competition_level}")
        
        if spec.sales_velocity not in cls.SALES_VELOCITY:
            raise ValueError(f"Invalid sales velocity: {spec.sales_velocity}")
        
        cls.SPECS[name] = spec 