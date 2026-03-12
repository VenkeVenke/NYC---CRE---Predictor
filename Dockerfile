# ============================================================
# Dockerfile — FastAPI Prediction Server
# ============================================================
#
# This builds a container image for the FastAPI app only.
# The MLflow server runs as a separate container (see docker-compose.yml).
#
# Build stages:
#   1. Start from official Python 3.11 slim image (smaller than full python)
#   2. Set working directory
#   3. Install Python dependencies
#   4. Copy source code
#   5. Expose port and define startup command
#
# Why Python 3.11 and not 3.13?
#   Render's free tier and most production environments use 3.11.
#   Pinning here ensures local Docker matches what runs in the cloud.
#   Our code is fully compatible with 3.11.

FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Install system dependencies needed by some Python packages
# (XGBoost needs libgomp for multi-threading on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — before source code.
# Why? Docker caches each layer. If requirements.txt hasn't changed,
# Docker reuses the cached pip install layer — much faster rebuilds.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
# (done after pip install so code changes don't invalidate the pip cache)
COPY src/ ./src/

# Copy the trained model file
# This is the fallback model used when MLflow is not available (e.g. on Render)
COPY models/ ./models/

# Expose the port FastAPI listens on
EXPOSE 8000

# Health check — Docker will ping this endpoint every 30s
# If it fails 3 times, the container is marked unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
# --host 0.0.0.0 → listen on all interfaces (required inside Docker)
# --port 8000    → match the EXPOSE above
# no --reload    → reload is for development only, not for containers
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
