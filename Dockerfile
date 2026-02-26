# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for caching)
COPY requirements-docker.txt .

# Install Python dependencies (no cache to save space)
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && rm -rf /root/.cache/pip

# Copy application code
COPY app.py .
COPY config.py .
COPY templates/ templates/
COPY static/ static/
COPY models/ models/
COPY src/ src/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run with gunicorn for production
# Railway provides PORT env variable
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 app:app
