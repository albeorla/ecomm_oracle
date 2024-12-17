from loguru import logger
import sys
from pathlib import Path
from contextlib import contextmanager
import time

# Configure default logger settings
DEFAULT_LOG_DIR = "logs"
DEFAULT_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
ERROR_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}\n{extra}"

def error_handler(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Uncaught exception:")

def setup_logger(log_dir: str = DEFAULT_LOG_DIR) -> None:
    """
    Set up loguru logger with file and console outputs.
    Overwrites logs on each run and includes rotation.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler and any existing handlers
    logger.remove()
    
    # Set up exception handling
    sys.excepthook = error_handler
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        format=DEFAULT_FORMAT,
        level="INFO",
        colorize=True,
        enqueue=True,
        catch=True,
        diagnose=True
    )
    
    # Add file handler for all logs
    logger.add(
        str(log_path / "app.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        mode="w",
        enqueue=True,
        catch=True,
        diagnose=True
    )
    
    # Add file handler for errors only
    logger.add(
        str(log_path / "error.log"),
        format=ERROR_FORMAT,
        level="ERROR",
        mode="w",
        enqueue=True,
        catch=True,
        backtrace=True,
        diagnose=True,
        filter=lambda record: record["level"].name in ("ERROR", "CRITICAL")
    )

    logger.info("Logger initialized")

@contextmanager
def log_execution_time(operation: str):
    """Context manager to log execution time of operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation} completed in {duration:.2f} seconds")

# Set up default logger on module import
setup_logger() 