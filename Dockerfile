# Backend — FastAPI (Python 3.11)
FROM python:3.11-slim

WORKDIR /app

# System libraries required by psycopg2 and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt tf-keras

# Copy application source
COPY backend/ backend/
COPY alembic.ini .

EXPOSE 8000

# Apply schema migrations on first boot and upgrades, then start the server
CMD ["sh", "-c", \
    "alembic upgrade head && \
     uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
