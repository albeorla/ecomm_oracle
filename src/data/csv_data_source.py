import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
from .data_source import DataSource
from .product_generator import ProductGenerator

class CSVDataSource(DataSource):
    """Data source implementation for CSV files with data generation capability."""
    
    def __init__(self, file_path: str = "data/products.csv", n_samples: int = 1000, seed: int = 42, product_type: str = "envelopes"):
        """
        Initialize the CSV data source.
        
        Args:
            file_path: Path to the CSV file
            n_samples: Number of samples to generate if file doesn't exist
            seed: Random seed for reproducibility when generating data
            product_type: Type of product to generate
        """
        logger.debug(f"Initializing CSVDataSource with path: {file_path}, samples: {n_samples}, seed: {seed}, product_type: {product_type}")
        self.file_path = Path(file_path)
        self.n_samples = n_samples
        self.seed = seed
        self.product_type = product_type
        self._data = None
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {self.file_path.parent}")
        
        # Generate data if file doesn't exist
        if not self.file_path.exists():
            logger.info(f"Data file not found at {self.file_path}, generating new data")
            self._generate_and_save_data()
        else:
            logger.debug(f"Found existing data file at {self.file_path}")
    
    def _generate_and_save_data(self) -> None:
        """Generate synthetic data and save it to CSV."""
        logger.info(f"Generating {self.n_samples} sample products for type: {self.product_type}")
        generator = ProductGenerator(seed=self.seed)
        df = generator.generate_products(self.n_samples, self.product_type)
        
        # Save to CSV
        logger.info(f"Saving generated data to {self.file_path}")
        try:
            df.to_csv(self.file_path, index=False)
            self._data = df
            logger.success(f"Successfully saved {len(df)} records to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {self.file_path}: {str(e)}")
            raise
    
    def _load_data(self) -> None:
        """Load data from CSV file if not already loaded."""
        if self._data is None:
            logger.debug(f"Loading data from {self.file_path}")
            try:
                self._data = pd.read_csv(self.file_path)
                logger.debug(f"Successfully loaded {len(self._data)} records")
                
                # Ensure required columns exist
                required_columns = {
                    'asin', 'category', 'subcategory', 'price', 'weight',
                    'review_rating', 'review_count', 'competitors',
                    'estimated_monthly_sales', 'fba_fees', 'cogs',
                    'monthly_profit', 'last_updated'
                }
                missing_columns = required_columns - set(self._data.columns)
                if missing_columns:
                    error_msg = f"Missing required columns: {missing_columns}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
            except Exception as e:
                logger.error(f"Error loading data from {self.file_path}: {str(e)}")
                raise
    
    def get_products(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve product data from the CSV file.
        
        Args:
            limit: Optional maximum number of records to retrieve
            
        Returns:
            DataFrame containing product data
        """
        logger.debug(f"Retrieving products with limit: {limit}")
        self._load_data()
        
        if limit is not None:
            logger.debug(f"Returning first {limit} records")
            return self._data.head(limit)
        
        logger.debug(f"Returning all {len(self._data)} records")
        return self._data
    
    def get_product_by_asin(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single product by its ASIN.
        
        Args:
            asin: The Amazon Standard Identification Number
            
        Returns:
            Dictionary containing product data or None if not found
        """
        logger.debug(f"Looking up product with ASIN: {asin}")
        self._load_data()
        
        product = self._data[self._data['asin'] == asin]
        if len(product) == 0:
            logger.warning(f"No product found with ASIN: {asin}")
            return None
        
        logger.debug(f"Found product with ASIN: {asin}")
        return product.iloc[0].to_dict() 