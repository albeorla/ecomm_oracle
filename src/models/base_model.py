"""Base model class for profitability prediction."""

import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class BaseModel:
    def __init__(self, model_dir: str = None):
        """Initialize the base model.
        
        Args:
            model_dir: Directory to save/load model files
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.model = None
        self.scaler = StandardScaler()
        
        # Define feature columns
        self.feature_columns = [
            'price',
            'weight',
            'review_rating',
            'review_count',
            'competitors',
            'estimated_monthly_sales',
            'fba_fees',
            'cogs',
            'seller_count',
            'bsr',
            'review_velocity',
            'seasonal_factor'
        ]
        
        # Define categorical columns for one-hot encoding
        self.categorical_columns = ['category']
    
    def preprocess_data(self, df):
        """Preprocess the data for training or prediction.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure all numeric features are float64
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        # Select numeric features
        X = df[self.feature_columns].copy()
        
        # One-hot encode categorical features if they exist
        for cat_col in self.categorical_columns:
            if cat_col in df.columns:
                # Get all possible categories (stored during first call)
                if not hasattr(self, 'categories_'):
                    self.categories_ = {
                        cat_col: sorted(df[cat_col].unique().tolist())
                    }
                
                # Create dummy columns with consistent ordering
                for category in self.categories_[cat_col]:
                    dummy_name = f"{cat_col}_{category}"
                    X[dummy_name] = (df[cat_col] == category).astype(np.float64)
        
        return X
    
    def train(self, df, tune_first: bool = False):
        """Train the model on the provided DataFrame.
        
        Args:
            df (pd.DataFrame): Training data
            tune_first (bool): Whether to tune hyperparameters before training
            
        Returns:
            dict: Training metrics
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, df):
        """Make predictions on new data.
        
        Args:
            df (pd.DataFrame): Data to make predictions on
            
        Returns:
            np.ndarray: Predicted values as a numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Preprocess and scale the features
        X = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions and ensure numpy array output
        predictions = self.model.predict(X_scaled)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        return predictions.astype(np.float64)
    
    def save(self, filename: str = 'model.joblib'):
        """Save the model to disk.
        
        Args:
            filename (str): Name of the file to save the model to
        """
        if self.model_dir is None:
            raise ValueError("model_dir not set")
            
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'categories_': getattr(self, 'categories_', {})
        }
        joblib.dump(model_data, model_path)
    
    def load(self, filename: str = 'model.joblib'):
        """Load the model from disk.
        
        Args:
            filename (str): Name of the file to load the model from
            
        Returns:
            BaseModel: Self for chaining
        """
        if self.model_dir is None:
            raise ValueError("model_dir not set")
            
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found at {model_path}")
            
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.categorical_columns = model_data['categorical_columns']
        if 'categories_' in model_data:
            self.categories_ = model_data['categories_']
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features (numpy array or DataFrame)
            y_test: Test target values (numpy array or Series)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # If X_test is already preprocessed and scaled (numpy array), use it directly
        if isinstance(X_test, np.ndarray):
            y_pred = self.model.predict(X_test)
        else:
            # If X_test is a DataFrame, preprocess and scale it
            X_test_processed = self.preprocess_data(X_test)
            X_test_scaled = self.scaler.transform(X_test_processed)
            y_pred = self.model.predict(X_test_scaled)
        
        # Convert y_test to numpy array if it isn't already
        if not isinstance(y_test, np.ndarray):
            y_test = y_test.to_numpy()
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    def prepare_data(self, df):
        """Prepare data for training by splitting into train/test sets.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) as numpy arrays
        """
        # Preprocess features
        X = self.preprocess_data(df)
        y = df['monthly_profit'].astype(np.float64)
        
        # Split the data
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to numpy arrays
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        return X_train_scaled, X_test_scaled, y_train, y_test