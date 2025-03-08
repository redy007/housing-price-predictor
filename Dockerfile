FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies with pinned versions to fix compatibility issues
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir werkzeug==2.0.3

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Create a proper stub model with fitted scaler and model
RUN echo 'import pickle' > create_stubs.py && \
    echo 'import numpy as np' >> create_stubs.py && \
    echo 'from sklearn.ensemble import RandomForestRegressor' >> create_stubs.py && \
    echo 'from sklearn.preprocessing import StandardScaler' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo '# Create synthetic data for fitting' >> create_stubs.py && \
    echo 'X_sample = np.array([' >> create_stubs.py && \
    echo '    [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252],' >> create_stubs.py && \
    echo '    [-122.22, 37.86, 21.0, 7099.0, 1106.0, 2401.0, 1138.0, 8.3014]' >> create_stubs.py && \
    echo '])' >> create_stubs.py && \
    echo 'y_sample = np.array([450000, 350000])  # Sample target values' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo '# Create and fit the scaler' >> create_stubs.py && \
    echo 'scaler = StandardScaler()' >> create_stubs.py && \
    echo 'X_scaled = scaler.fit_transform(X_sample)' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo '# Create and fit the model' >> create_stubs.py && \
    echo 'model = RandomForestRegressor(n_estimators=1, random_state=42)' >> create_stubs.py && \
    echo 'model.fit(X_scaled, y_sample)' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo '# Save the fitted scaler' >> create_stubs.py && \
    echo 'with open("models/scaler.pkl", "wb") as f:' >> create_stubs.py && \
    echo '    pickle.dump(scaler, f)' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo '# Save the fitted model' >> create_stubs.py && \
    echo 'with open("models/model.pkl", "wb") as f:' >> create_stubs.py && \
    echo '    pickle.dump(model, f)' >> create_stubs.py && \
    echo '' >> create_stubs.py && \
    echo 'print("Created and fitted stub model files successfully")' >> create_stubs.py && \
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