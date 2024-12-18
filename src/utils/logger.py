from loguru import logger
import sys
from pathlib import Path
from contextlib import contextmanager
import time
import traceback

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

def validate_error_logging():
    """Test error logging to ensure it's working properly."""
    try:
        raise ValueError("Test error for logging validation")
    except Exception as e:
        logger.error(
            "Logging validation test",
            exc_info=True,
            extra={"test": "validation"}
        )
        return True
    return False

def setup_logger(
    app_log_path: str = "logs/app.log",
    error_log_path: str = "logs/error.log"
) -> None:
    """
    Configure application logging with proper error handling.
    
    Args:
        app_log_path: Path for general application logs
        error_log_path: Path for error-specific logs
    """
    # Create log directories
    for path in [app_log_path, error_log_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing handlers
    logger.remove()
    
    # Add console handler for INFO and above
    logger.add(
        sink=sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler for ERROR and above with extended debug info
    logger.add(
        sink=error_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message} | {extra}\n{exception}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
        catch=True,
        enqueue=True,
        rotation="1 week",
        retention="1 month",
        compression="zip"
    )
    
    # Add app.log handler for all levels
    logger.add(
        sink=app_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 week",
        retention="1 month",
        compression="zip",
        enqueue=True
    )
    
    # Intercept uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    # Set up the exception handler
    sys.excepthook = handle_exception
    
    # Validate error logging is working
    if validate_error_logging():
        logger.info("Logger initialized and validated")
    else:
        print("WARNING: Logger validation failed!", file=sys.stderr)

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