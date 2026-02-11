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

    # OpenAI
    OPENAI_API_KEY: str

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
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    RATE_LIMIT: str = "10/minute"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
