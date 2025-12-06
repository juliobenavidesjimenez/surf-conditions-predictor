"""
Preprocessing and feature engineering module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_time_features(df):
    """
    Extract time-based features from datetime column
    
    Parameters:
        df: DataFrame with 'time' column as datetime
        
    Returns:
        DataFrame with new time features
    """
    df = df.copy()  # Don't modify original
    
    # Extract temporal features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    
    return df

def create_wave_features(df):
    """
    Create derived features from wave variables
    
    Parameters:
        df: DataFrame with wave data
        
    Returns:
        DataFrame with new wave features
    """
    df = df.copy()
    
    # Ratio features
    df['wave_steepness'] = df['wave_height'] / df['wave_period']
    df['wind_ratio'] = df['wind_wave_height'] / (df['wave_height'] + 0.01)  # +0.01 to avoid division by zero
    
    return df

def preprocess_data(df):
    """
    Complete preprocessing pipeline
    
    Parameters:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Apply feature engineering
    df = create_time_features(df)
    df = create_wave_features(df)
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
        df: DataFrame with features and target
        test_size: proportion for test set (default 0.2 = 20%)
        random_state: seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(['conditions', 'time'], axis=1)
    y = df['conditions']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    Fit on train, transform both train and test
    
    Parameters:
        X_train: training features
        X_test: test features
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    # Fit only on train (important!)
    scaler.fit(X_train)
    
    # Transform both
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler