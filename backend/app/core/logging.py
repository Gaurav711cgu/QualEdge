"""
Structured JSON logger for production telemetry.
"""
import json
import logging
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Format standard log records into structured JSON payloads.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "logger": record.name
        }
        # Add correlation ID if present (useful for microservices request tracing)
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            log_payload["correlation_id"] = correlation_id
            
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_payload)

def setup_logging(level=logging.INFO):
    """Configures structured JSON logging for all active loggers."""
    root_logger = logging.getLogger()
    
    # Remove existing default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(level)

    # Sanitize uvicorn logs to match structured formatting
    for uvicorn_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        u_logger = logging.getLogger(uvicorn_logger_name)
        u_logger.handlers = []
        u_logger.propagate = True
