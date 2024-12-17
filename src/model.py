import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class ProfitabilityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'price', 'weight', 'review_rating', 'review_count',
            'competitors', 'estimated_monthly_sales', 'fba_fees', 'cogs'
        ]
    
    def preprocess_data(self, df):
        """Preprocess the data for training or prediction."""
        # One-hot encode categorical variables
        X = pd.get_dummies(df[self.feature_columns], columns=['category'])
        return X
    
    def train(self, df):
        """Train the model on the provided DataFrame."""
        X = self.preprocess_data(df)
        y = df['monthly_profit']
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on validation set
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        return metrics
    
    def predict(self, df):
        """Make predictions on new data."""
        X = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) 