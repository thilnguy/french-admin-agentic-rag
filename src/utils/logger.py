import sys
import logging
from typing import Optional

def setup_logger(name: str = "french_admin_agent", level: Optional[str] = None):
    """
    Configure a structured logger for the application.
    """
    logger = logging.getLogger(name)
    
    # Set log level (default to INFO if not specified)
    log_level = level or "INFO"
    logger.setLevel(log_level)
    
    # Prevent adding multiple handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger
        
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Create formatter
    # For production, we might want JSON formatting. For now, a clean standard format.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()
