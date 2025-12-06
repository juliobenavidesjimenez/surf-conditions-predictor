"""
Train and evaluate wave height prediction models (Regression)
"""

import pandas as pd
from src.preprocessing import preprocess_data, split_data_regression, scale_features
from src.model import (
    train_linear_regression, 
    train_random_forest_regressor,
    evaluate_regression_model
)

# Load data
print("Loading data...")
df = pd.read_csv("data/raw/zarautz_2024-01-01_2024-12-05.csv")
df['time'] = pd.to_datetime(df['time'])

print(f"Dataset shape: {df.shape}")
print(f"Wave height stats:\n{df['wave_height'].describe()}")

# Preprocessing
print("\nPreprocessing...")
df = preprocess_data(df)

# Split (no need for conditions now)
X_train, X_test, y_train, y_test = split_data_regression(df)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Train Linear Regression
print("\n" + "="*60)
lr_model = train_linear_regression(X_train_scaled, y_train)
evaluate_regression_model(lr_model, X_test_scaled, y_test, "Linear Regression")

# Train Random Forest
print("\n" + "="*60)
rf_model = train_random_forest_regressor(X_train_scaled, y_train)
evaluate_regression_model(rf_model, X_test_scaled, y_test, "Random Forest Regressor")

print("\n" + "="*60)
print("✅ Training complete!")