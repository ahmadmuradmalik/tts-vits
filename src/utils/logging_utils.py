import logging
import sys
from pathlib import Path

def setup_logging(log_file_path: Path, level: int = logging.INFO):
    """
    Configures logging to write to a file and to stdout.

    Args:
        log_file_path: Path to the log file.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs if this is called multiple times
    # (though typically it should only be called once)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s' # Simpler format for console
    )
    console_handler.setFormatter(console_formatter)
    # Only add console handler if no other console handlers are present (e.g. from basicConfig)
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in root_logger.handlers):
        root_logger.addHandler(console_handler)

    logging.info(f"Logging configured. Log file: {log_file_path}")

if __name__ == '__main__':
    # Example usage:
    setup_logging(Path('./example.log'))
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.") 