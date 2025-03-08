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
