
# Use the same Python version as our development environment.
FROM python:3.14.6-slim

# Prevent Python from generating unnecessary cache files.
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure application logs appear immediately in container output.
ENV PYTHONUNBUFFERED=1

# Set the application working directory.
WORKDIR /app

# Install application dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source code.
COPY src/ ./src/

# Copy the trained model artifact.
COPY artifacts/random_forest_candidate.joblib ./artifacts/random_forest_candidate.joblib

# Create the directory used for prediction logs.
RUN mkdir -p /app/logs

# Expose the API port.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
