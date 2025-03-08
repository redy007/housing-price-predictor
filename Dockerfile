FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies with specific versions to avoid compatibility issues
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir werkzeug==2.0.3

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Create stub model files - using a separate file approach
RUN echo 'import pickle' > create_stubs.py && \
    echo 'import numpy as np' >> create_stubs.py && \
    echo 'from sklearn.ensemble import RandomForestRegressor' >> create_stubs.py && \
    echo 'from sklearn.preprocessing import StandardScaler' >> create_stubs.py && \
    echo 'model = RandomForestRegressor(n_estimators=1)' >> create_stubs.py && \
    echo 'scaler = StandardScaler()' >> create_stubs.py && \
    echo 'with open("models/model.pkl", "wb") as f:' >> create_stubs.py && \
    echo '    pickle.dump(model, f)' >> create_stubs.py && \
    echo 'with open("models/scaler.pkl", "wb") as f:' >> create_stubs.py && \
    echo '    pickle.dump(scaler, f)' >> create_stubs.py && \
    echo 'print("Created stub model files successfully")' >> create_stubs.py && \
    python create_stubs.py

# Set environment variables
ENV MODEL_PATH=models/model.pkl
ENV SCALER_PATH=models/scaler.pkl
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 5000

# Use a script to start the app
RUN echo '#!/bin/bash' > start.sh && \
    echo 'echo "Starting Flask application..."' >> start.sh && \
    echo 'pip list' >> start.sh && \
    echo 'python app/api.py' >> start.sh && \
    chmod +x start.sh

# Command to run the application
CMD ["./start.sh"]