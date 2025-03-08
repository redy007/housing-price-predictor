import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and scaler
model_path = os.environ.get('MODEL_PATH', 'models/model.pkl')
scaler_path = os.environ.get('SCALER_PATH', 'models/scaler.pkl')

# Check if model files exist at startup
if not os.path.exists(model_path):
    print(f"WARNING: Model file not found at {model_path}")
    print(f"Current directory contents: {os.listdir('.')}")
    print(f"Models directory contents: {os.listdir('models/') if os.path.exists('models/') else 'models/ not found'}")

if not os.path.exists(scaler_path):
    print(f"WARNING: Scaler file not found at {scaler_path}")

# Load model and scaler
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    # Set placeholder for testing
    model = None
    scaler = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model is not None, "scaler_loaded": scaler is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.json
        
        # Check if we have valid model and scaler
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500
        
        # Convert to feature array (ensure proper ordering)
        features = np.array([[
            float(data.get('longitude', 0)),
            float(data.get('latitude', 0)),
            float(data.get('housing_median_age', 0)),
            float(data.get('total_rooms', 0)),
            float(data.get('total_bedrooms', 0)),
            float(data.get('population', 0)),
            float(data.get('households', 0)),
            float(data.get('median_income', 0))
        ]])
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            "prediction": float(prediction),
            "units": "USD"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # When run directly, bind to all interfaces
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
