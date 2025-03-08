from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load the model and scaler paths from environment variables
model_path = os.environ.get('MODEL_PATH', 'models/model.pkl')
scaler_path = os.environ.get('SCALER_PATH', 'models/scaler.pkl')

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
    
    return jsonify({'prediction': float(prediction)})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
