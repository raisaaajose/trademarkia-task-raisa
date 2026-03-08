FROM python:3.11-slim

WORKDIR /app

#install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Pre-download the model weights so it doesn't happen at runtime
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy code and the pre-processed data folder
COPY data/ ./data/
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app

EXPOSE 8000

# Start the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]