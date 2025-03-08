import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the housing dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the housing data."""
    # Handle missing values
    df = df.dropna()

    # Drop ocean_proximity to avoid float conversion errors
    if 'ocean_proximity' in df.columns:
        df = df.drop('ocean_proximity', axis=1)
    
    # Select features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_scaler(scaler, filepath='models/scaler.pkl'):
    """Save the fitted scaler."""
    import joblib
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
