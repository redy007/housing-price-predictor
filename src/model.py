from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(X_train, y_train):
    """Train a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='models/model.pkl'):
    """Save the trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath='models/model.pkl'):
    """Load a trained model."""
    return joblib.load(filepath)
