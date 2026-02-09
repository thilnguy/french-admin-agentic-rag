# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies first for caching
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Pre-download BGE-M3 model to avoid runtime download
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-m3')"

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY . .

# Environment variables
ENV QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    PYTHONPATH=/app

EXPOSE 8000

# Run using Uvicorn with multiple workers for production
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
