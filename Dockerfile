# Dockerfile for inference-sentinel
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    httpx>=0.26.0 \
    structlog>=24.1.0 \
    prometheus-client>=0.19.0 \
    opentelemetry-api>=1.22.0 \
    opentelemetry-sdk>=1.22.0 \
    opentelemetry-exporter-otlp>=1.22.0 \
    opentelemetry-instrumentation-fastapi>=0.43b0 \
    pyyaml>=6.0 \
    anthropic>=0.18.0 \
    google-generativeai>=0.4.0 \
    numpy>=1.24.0 \
    cachetools>=5.3.0 \
    transformers>=4.37.0 \
    torch>=2.2.0 \
    sentencepiece>=0.1.99

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/

EXPOSE 8000

CMD ["uvicorn", "sentinel.main:app", "--host", "0.0.0.0", "--port", "8000"]