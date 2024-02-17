#!/usr/bin/env python

import glob
import numpy as np
import os
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Load and preprocess data
def load_data(directory="data"):
    file_pattern = os.path.join(directory, 'products*.csv')
    csv_files = glob.glob(file_pattern)
    concatenated_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    return concatenated_df


def preprocess_features(data):
    # Convert 'Price', 'Reviews', 'Rating', and 'Net' to numeric types, removing '$' and ','.
    for col in ['Price', 'Reviews', 'Rating', 'Net']:
        if col in data.columns:
            # Ensure the column is treated as a string before applying string methods
            data[col] = data[col].astype(str).str.replace('[$,]', '', regex=True)
            # Now safely convert to numeric, as all values are guaranteed to be string representations
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Calculate 'ProductAge' and other engineered features here as needed
    if 'Date First Available' in data.columns:
        data['Date First Available'] = pd.to_datetime(data['Date First Available'], errors='coerce')
        data['ProductAge'] = (pd.to_datetime('now') - data['Date First Available']).dt.days

    return data

def preprocess_data(features):
    # Identify numeric and categorical features
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features.select_dtypes(include=['object']).columns.tolist()

    # Remove target variable 'Net' if it's mistakenly included in numeric_features
    if 'Net' in numeric_features:
        numeric_features.remove('Net')

    # Define preprocessing steps for numeric and categorical features
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    return preprocessor


def perform_grid_search(pipeline, X_train, y_train):
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6],
        'model__min_samples_split': [2, 4, 6],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    logger.info(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    logger.info(f"MSE: {mean_squared_error(y_test, y_pred)}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    logger.info(f"R² : {r2_score(y_test, y_pred)}")
    logger.info(f"MedAE: {median_absolute_error(y_test, y_pred)}")


def main():
    data = load_data()
    data = preprocess_features(data)  # Direct conversions and feature engineering

    # Separate features and target variable
    features = data.drop(columns=['Net', 'ASIN'])
    labels = data['Net'].dropna()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Setup the preprocessing pipeline
    preprocessor = preprocess_data(features)  # Setup preprocessing steps
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', GradientBoostingRegressor(random_state=42))])

    # Perform grid search and model evaluation
    grid_search = perform_grid_search(pipeline, X_train, y_train)
    evaluate_model(grid_search.best_estimator_, X_test, y_test)



if __name__ == '__main__':
    main()
