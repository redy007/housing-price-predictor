import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

# Create Flask app
app = Flask(__name__)

# Setup simple logging
@app.before_first_request
def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    app.logger.info("Flask application started")

# Global variables for model and scaler
model = None
scaler = None

# Basic route to verify the server is running
@app.route('/', methods=['GET'])
def root():
    return "Housing Price Predictor API is running!"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Attempt to load model and scaler on first health check
        global model, scaler
        
        if model is None or scaler is None:
            model_path = os.environ.get('MODEL_PATH', 'models/model.pkl')
            scaler_path = os.environ.get('SCALER_PATH', 'models/scaler.pkl')
            
            app.logger.info(f"Loading model from {model_path}")
            app.logger.info(f"Loading scaler from {scaler_path}")
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                app.logger.info("Model and scaler loaded successfully")
            except Exception as e:
                app.logger.error(f"Error loading model or scaler: {str(e)}")
        
        return jsonify({
            "status": "ok", 
            "model_loaded": model is not None, 
            "scaler_loaded": scaler is not None
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        app.logger.info(f"Received prediction request: {data}")
        
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500
        
        # Convert to feature array
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
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
