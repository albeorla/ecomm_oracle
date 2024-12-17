from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any

class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    @abstractmethod
    def get_products(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve product data from the source.
        
        Args:
            limit: Optional maximum number of records to retrieve
            
        Returns:
            DataFrame containing product data
        """
        pass
    
    @abstractmethod
    def get_product_by_asin(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single product by its ASIN.
        
        Args:
            asin: The Amazon Standard Identification Number
            
        Returns:
            Dictionary containing product data or None if not found
        """
        pass 