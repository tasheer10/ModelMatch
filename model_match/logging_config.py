import logging
import sys
import os

# Define standard log formats
LOG_FORMAT_SIMPLE = "%(levelname)s: %(message)s"
LOG_FORMAT_DETAILED = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: int = logging.INFO, log_format: str = LOG_FORMAT_DETAILED):
    """
    Configures the root logger for the application.

    Args:
        level: The minimum logging level to output (e.g., logging.DEBUG, logging.INFO).
        log_format: The format string for log messages.
    """
    # Get the root logger
    logger = logging.getLogger() # Get the root logger
    # Or, get a specific logger for your application:
    # logger = logging.getLogger("llm_comparator")

    # Prevent adding multiple handlers if called multiple times (e.g., in tests)
    if logger.hasHandlers():
        # Clear existing handlers if you want to reconfigure completely
        # logger.handlers.clear()
        # Or simply return if already configured
        return

    # Set the logger's level. This acts as a global filter.
    logger.setLevel(level)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout) # Log to standard output
    # Set the handler's level (it can be different from the logger's level)
    # For simplicity, we'll use the same level for now.
    console_handler.setLevel(level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # --- Optional: File Handler (Example) ---
    # You might want to add file logging later:
    # log_dir = "logs"
    # os.makedirs(log_dir, exist_ok=True)
    # file_handler = logging.FileHandler(os.path.join(log_dir, "comparator.log"))
    # file_handler.setLevel(logging.DEBUG) # Log more details to the file
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # You can also adjust levels for noisy libraries if needed
    # logging.getLogger("openai").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.INFO)

    logger.debug("Logging setup complete.") # This will only show if level is DEBUG

# Example of getting a specific logger instance for a module
# You would typically do this within each module like:
# import logging
# logger = logging.getLogger(__name__)