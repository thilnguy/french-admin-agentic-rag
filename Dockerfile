# Stage 1: Builder
FROM python:3.13-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy only dependency files first
COPY pyproject.toml uv.lock ./

# Install dependendencies without installing the project itself yet
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download models to cache them
# We activate the venv and run python to download the model
# RUN . .venv/bin/activate && python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-m3')"

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/home/appuser/.cache/huggingface"

WORKDIR /app

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy model cache
# Copy model cache
# COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application code
COPY --from=builder /app /app

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables defaults
ENV PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
