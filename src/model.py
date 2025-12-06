"""
Machine Learning models for wave height prediction (Regression)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model
    
    Parameters:
        X_train: training features
        y_train: training target (wave heights)
        
    Returns:
        trained model
    """
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    return model


def train_random_forest_regressor(X_train, y_train):
    """
    Train Random Forest Regressor
    
    Parameters:
        X_train: training features
        y_train: training target
        
    Returns:
        trained model
    """
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    return model


def evaluate_regression_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate regression model
    
    Parameters:
        model: trained model
        X_test: test features
        y_test: test target
        model_name: name for display
    """
    print(f"\n=== {model_name} Evaluation ===")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE (Mean Absolute Error): {mae:.3f} meters")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f} meters")
    print(f"R² Score: {r2:.3f}")
    
    # Sample predictions
    print("\nSample Predictions:")
    print("Real    Predicted")
    for i in range(5):
        print(f"{y_test.iloc[i]:.2f}m   {y_pred[i]:.2f}m")