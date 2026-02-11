import sys
import logging
import json
from datetime import datetime, timezone
from typing import Optional
from src.config import settings


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)


def setup_logger(name: str = "french_admin_agent", level: Optional[str] = None):
    """
    Configure a structured logger for the application.
    Uses JSON format in production, plain text in debug mode.
    """
    logger = logging.getLogger(name)

    log_level = level or settings.LOG_LEVEL
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Use JSON in production, plain text in debug
    if settings.DEBUG:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = JSONFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Create a default logger instance
logger = setup_logger()
