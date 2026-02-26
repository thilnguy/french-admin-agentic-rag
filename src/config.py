from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """
    Centralized application configuration using Pydantic Settings.
    Reads from environment variables and .env file.
    """

    # App Config
    APP_NAME: str = "French Admin Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # OpenAI / Local LLM
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    GUARDRAIL_MODEL: str = "gpt-4o-mini"  # Model used for topic validation and hallucination checks
    FAST_LLM_MODEL: str = "gpt-4o-mini"  # Model used for lightweight tasks (query rewriting, intent classification)
    LLM_PROVIDER: str = "openai"  # "openai" or "local"
    LOCAL_LLM_URL: str = "http://localhost:8000/v1"
    LOCAL_LLM_MODEL: str = "qwen-7b-french-admin"


    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None

    # Security
    ALLOWED_ORIGINS: str = "*"
    RATE_LIMIT: str = "10/minute"
    API_KEY: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

    # OpenTelemetry Tracing
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4318/v1/traces"
    OTEL_SERVICE_NAME: str = "french-admin-agent"

    # Legal Data Update Pipeline (HuggingFace)
    HF_DATASET_NAME: str = "your-hf-username/french-legal-data"
    HUGGINGFACE_TOKEN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings():
    try:
        return Settings()
    except PermissionError:
        # Fallback if .env is inaccessible (e.g. during CI/constrained environments)
        # Pydantic will still use environment variables
        class SafeSettings(Settings):
            model_config = SettingsConfigDict(env_file=None)

        return SafeSettings()


settings = get_settings()
