h1. Student project - sda 

---
# CI/CD for Data Science: A Beginner's Guide

This guide demonstrates how to implement Continuous Integration and Continuous Deployment (CI/CD) for a simple data science project in Python.

## Project Overview

We'll create a simple machine learning model that predicts housing prices. The CI/CD pipeline will:

1. Run tests automatically when code is pushed
2. Check code quality and style
3. Build and test the ML model
4. Deploy the model to a production-like environment

## Project Structure

```
housing-price-predictor/
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── data/
│   └── housing.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   └── test_model.py
├── app/
│   ├── __init__.py
│   └── api.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## 1. Setting Up the Data Science Code

Let's start with the core data science code. These files handle data processing, model training, and evaluation.

### data_processing.py
```python
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
```

### model.py
```python
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
```

### evaluate.py
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2
    }
```

## 2. Creating Tests

Testing is a critical part of CI/CD. Let's write some basic tests.

### test_data_processing.py
```python
import unittest
import pandas as pd
import numpy as np
import os
from src.data_processing import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'longitude': [-122.23, -122.22],
            'latitude': [37.88, 37.86],
            'housing_median_age': [41.0, 21.0],
            'total_rooms': [880.0, 7099.0],
            'total_bedrooms': [129.0, 1106.0],
            'population': [322.0, 2401.0],
            'households': [126.0, 1138.0],
            'median_income': [8.3252, 8.3014],
            'median_house_value': [452600.0, 358500.0]
        })
        
        # Save the sample data for testing load_data
        os.makedirs('data', exist_ok=True)
        self.sample_data.to_csv('data/test_housing.csv', index=False)
    
    def test_load_data(self):
        """Test if data is loaded correctly."""
        df = load_data('data/test_housing.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), list(self.sample_data.columns))
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X_train, X_test, y_train, y_test, scaler = preprocess_data(self.sample_data)
        
        # Check shapes
        self.assertEqual(X_train.shape[1], 8)  # 8 features
        self.assertEqual(len(y_train) + len(y_test), 2)  # 2 samples total
        
        # Check if data is scaled
        self.assertTrue(np.abs(X_train.mean(axis=0)).max() < 1e-10)
        
    def tearDown(self):
        if os.path.exists('data/test_housing.csv'):
            os.remove('data/test_housing.csv')

if __name__ == '__main__':
    unittest.main()
```

### test_model.py
```python
import unittest
import numpy as np
from src.model import train_model
from src.evaluate import evaluate_model

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 8)
        self.y_train = 5 * self.X_train[:, 0] + 3 * self.X_train[:, 1] + np.random.normal(0, 0.1, 100)
        self.X_test = np.random.rand(20, 8)
        self.y_test = 5 * self.X_test[:, 0] + 3 * self.X_test[:, 1] + np.random.normal(0, 0.1, 20)
    
    def test_train_model(self):
        """Test model training."""
        model = train_model(self.X_train, self.y_train)
        
        # Check if model has been trained
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        model = train_model(self.X_train, self.y_train)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        # Check if metrics are calculated
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Check if metrics are reasonable
        self.assertGreater(metrics['r2'], 0)  # R² should be positive for a decent model

if __name__ == '__main__':
    unittest.main()
```

## 3. Creating an API for Model Deployment

### api.py
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model_path = os.environ.get('MODEL_PATH', 'models/model.pkl')
scaler_path = os.environ.get('SCALER_PATH', 'models/scaler.pkl')

# These will be loaded when the app starts
model = None
scaler = None

@app.before_first_request
def load_model_and_scaler():
    global model, scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    data = request.json
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return jsonify({
        'prediction': float(prediction)
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 4. Setting Up Docker for Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make the directory for models
RUN mkdir -p models

# Environment variables for model paths
ENV MODEL_PATH=models/model.pkl
ENV SCALER_PATH=models/scaler.pkl

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app/api.py"]
```

### requirements.txt
```
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
flask==2.0.1
joblib==1.0.1
pytest==6.2.5
```

## 5. Setting Up CI/CD with GitHub Actions

### ci_cd.yml
```yaml
name: CI/CD Pipeline for Data Science

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Check code style
      run: |
        pip install flake8
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
  
  build-and-train:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Download sample dataset
      run: |
        mkdir -p data
        wget https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv -O data/housing.csv
    
    - name: Train model
      run: |
        # Create a training script
        echo '
        import pandas as pd
        from src.data_processing import load_data, preprocess_data, save_scaler
        from src.model import train_model, save_model
        from src.evaluate import evaluate_model
        
        # Load and preprocess data
        df = load_data("data/housing.csv")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Train and save model
        model = train_model(X_train, y_train)
        save_model(model)
        save_scaler(scaler)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Model RMSE: {metrics[\"rmse\"]}")
        print(f"Model R²: {metrics[\"r2\"]}")
        ' > train.py
        
        # Run the training script
        python train.py
    
    - name: Save model and scaler
      uses: actions/upload-artifact@v2
      with:
        name: models
        path: models/
  
  build-docker:
    needs: build-and-train
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Download models
      uses: actions/download-artifact@v2
      with:
        name: models
        path: models/
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        push: false  # In a real scenario, you would push to a registry
        tags: housing-price-predictor:latest
    
    - name: Test Docker image
      run: |
        docker run -d -p 5000:5000 --name predictor housing-price-predictor:latest
        sleep 5  # Wait for container to start
        
        # Test health endpoint
        response=$(curl -s http://localhost:5000/health)
        echo "Health check response: $response"
        
        # Test prediction endpoint
        curl -X POST -H "Content-Type: application/json" -d '{
          "longitude": -122.23,
          "latitude": 37.88,
          "housing_median_age": 41.0,
          "total_rooms": 880.0,
          "total_bedrooms": 129.0,
          "population": 322.0,
          "households": 126.0,
          "median_income": 8.3252
        }' http://localhost:5000/predict
```

## 6. Setting Up a Main Script

Let's create a simple script to train and save our model. This will be useful for local development and testing.

### train.py
```python
import argparse
import pandas as pd
from src.data_processing import load_data, preprocess_data, save_scaler
from src.model import train_model, save_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train housing price prediction model')
    parser.add_argument('--data', type=str, default='data/housing.csv',
                        help='Path to the housing dataset')
    args = parser.parse_args()

    # Load and preprocess data
    print("Loading data...")
    df = load_data(args.data)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Save model and scaler
    print("Saving model and scaler...")
    save_model(model)
    save_scaler(scaler)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"Model RMSE: {metrics['rmse']:.2f}")
    print(f"Model R²: {metrics['r2']:.2f}")

if __name__ == '__main__':
    main()
```

## Teaching Guide

### 1. Introduction to CI/CD for Data Science

**CI/CD Core Concepts:**
- **Continuous Integration (CI)**: Automatically testing code changes to detect issues early
- **Continuous Deployment (CD)**: Automatically deploying approved changes to production

**Why CI/CD is Crucial for Data Science:**
- Ensures model reproducibility
- Maintains code quality
- Automates testing of data pipelines
- Streamlines model deployment
- Enables model versioning

### 2. Teaching Steps

1. **Setup GitHub Repository**
   - Create a new repo and upload the project structure
   - Explain each file's purpose

2. **Explain the Data Science Workflow**
   - Data loading and preprocessing
   - Model training and evaluation
   - Model saving and deployment
   - Testing at each stage

3. **Walk Through the CI/CD Pipeline**
   - Discuss the GitHub Actions workflow
   - Show how tests run automatically on push
   - Demonstrate how model training is automated
   - Explain how the Docker container is built and tested

4. **Run a Demo**
   - Make a code change (e.g., add a new feature)
   - Push to GitHub and watch the pipeline run
   - Show how tests catch issues
   - Demonstrate the deployed model via the API

### 3. Key Learning Points

- **Data Science Pipeline Automation**
  - How to automate ML model training
  - How to ensure data preprocessing is consistent

- **Testing for Data Science**
  - How to test data processing code
  - How to test model performance
  - How to test model deployment

- **Containerization for ML Models**
  - How to package models in Docker
  - How to deploy models as APIs

- **CI/CD Implementation**
  - How to set up GitHub Actions for data science
  - How to automate the entire workflow

### 4. Exercise Ideas

1. **Add a New Feature**
   - Have students add a new feature to the model
   - Push changes and observe how CI/CD validates them

2. **Create a Model Drift Check**
   - Add monitoring for model performance degradation
   - Integrate it into the CI/CD pipeline

3. **Extend the API**
   - Add new endpoints (e.g., batch prediction)
   - Update tests and observe CI/CD in action

4. **Implement Model Versioning**
   - Add a model registry
   - Update the CI/CD pipeline to track versions

## Conclusion

This demo project provides a practical introduction to CI/CD for data science. It demonstrates how to automate testing, model training, and deployment using GitHub Actions and Docker. By following this guide, students will understand how to implement robust, automated workflows for their data science projects.
