FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Add debugging tools
RUN apt-get update && apt-get install -y curl procps net-tools && apt-get clean

# Environment variables for model paths and debugging
ENV MODEL_PATH=models/model.pkl
ENV SCALER_PATH=models/scaler.pkl 
ENV FLASK_APP=app/api.py
ENV PYTHONUNBUFFERED=1

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

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application - using direct Python execution
CMD ["python", "app/api.py"]