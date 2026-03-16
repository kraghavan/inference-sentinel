# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Builder stage
FROM base AS builder

# Install build dependencies
RUN pip install --upgrade pip hatchling

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# Runtime stage
FROM base AS runtime

# Create non-root user
RUN useradd --create-home --shell /bin/bash sentinel
USER sentinel
WORKDIR /home/sentinel/app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install the package
RUN pip install --user /wheels/*.whl

# Add local bin to PATH
ENV PATH="/home/sentinel/.local/bin:$PATH"

# Copy config directory
COPY --chown=sentinel:sentinel config/ config/

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "sentinel.main"]
