import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Define the base log structure
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include any extra attributes passed to the logger
        if hasattr(record, "audit_data"):
            log_entry["audit_data"] = record.audit_data
            
        return json.dumps(log_entry)

def get_audit_logger(name: str = "audit_logger") -> logging.Logger:
    """
    Returns a configured audit logger that outputs structured JSON to logs/audit.log.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured to avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Keep up to 10 backups of 10MB each
        handler = RotatingFileHandler(
            filename="logs/audit.log", 
            maxBytes=10 * 1024 * 1024, 
            backupCount=10,
            encoding="utf-8"
        )
        
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent logs from propagating to the root logger which might write to stdout
        logger.propagate = False
        
    return logger

# Global audit logger instance
audit_logger = get_audit_logger()
