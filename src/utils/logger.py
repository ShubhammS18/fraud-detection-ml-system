import logging
import json
import sys
from datetime import datetime

from numpy import record

class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format for industry standard tools."""
    def format(self, record):
            # Basic log structure
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            }
        
        # Standard logging.info(msg, extra={...}) merges keys into the record dict.
        # We extract keys that aren't part of the standard LogRecord attributes.
        standard_attrs = {
            'args', 'asctime', 'created', 'exc_info', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
            'msg', 'name', 'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'thread', 'threadName'
            }
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        return json.dumps(log_record)
        
def setup_logger(name="fraud_detection"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    
    # File Handler
    file_handler = logging.FileHandler("logs/api.log")
    file_handler.setFormatter(JSONFormatter())

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# Initialize once
logger = setup_logger()