#!/usr/bin/env python

import glob
import numpy as np
import os
import pandas as pd
import wandb
from inference.env_config import EnvConfig
from joblib import dump
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
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


def perform_grid_search(pipeline, x_train, y_train):
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
    grid_search.fit(x_train, y_train)

    logger.info(f"Best parameters found: {grid_search.best_params_}")
    wandb.log({"best_params": grid_search.best_params_,
               "best_score": grid_search.best_score_})
    return grid_search


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MedAE": median_absolute_error(y_test, y_pred)
    }
    logger.info(f"Metrics: {metrics}")
    wandb.log(metrics)


def main():
    config = EnvConfig()
    np.random.seed(config.random_seed)


    wandb.init(project="ecomm-oracle", config=config.__dict__,
               name="GradientBoostingRegressor-Experiment", tags=["GBR", "grid-search"],
               notes="Running grid search on Gradient Boosting Regressor with e-commerce data.")

    data = load_data()

    # Direct conversions and feature engineering
    data = preprocess_features(data)

    # Save preprocessed data to a CSV file
    data.to_csv(config.preprocessed_data_path, index=False)

    # Log the preprocessed dataset as an artifact
    artifact = wandb.Artifact('preprocessed_dataset', type='dataset')
    artifact.add_file(config.preprocessed_data_path)
    wandb.log_artifact(artifact)

    # Separate features and target variable
    features = data.drop(columns=['Net', 'ASIN'])
    labels = data['Net'].dropna()

    # Assuming config has attributes relevant to the model training
    wandb.config.update({
        "test_size": 0.2,
        "random_state": config.random_seed,
        "model": "GradientBoostingRegressor",
        "preprocessing": {
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "most_frequent",
            "encoder": "OneHotEncoder",
            "scaler": "StandardScaler"
        }
    })

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=config.random_seed
    )

    # Setup preprocessing steps
    preprocessor = preprocess_data(features)

    artifact = wandb.Artifact('dataset', type='dataset')
    artifact.add_dir('data')
    wandb.log_artifact(artifact)

    # Set up the preprocessing pipeline
    if config.run_pipeline:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', GradientBoostingRegressor(random_state=config.random_seed))])

        # Perform grid search and model evaluation
        grid_search = perform_grid_search(pipeline, x_train, y_train)
        evaluate_model(grid_search.best_estimator_, x_test, y_test)

        # Save the trained model to disk
        dump(grid_search.best_estimator_, 'model.pkl')

        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file('model.pkl')  # Now the model is saved and can be added as an artifact
        wandb.log_artifact(model_artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
