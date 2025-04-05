import datetime
import logging
import pathlib


def setup_logger():
    """Set up and return a logger that writes to both console and a file."""
    # Create timestamp for the log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log directory
    log_dir = f"exps/{timestamp}"
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configure logger
    log_file = f"{log_dir}/main.log"

    # Create logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger


def setup_child_logger(logger_name, parent_logger=None, subfolder=""):
    """
    Set up a child logger that writes to its own file.

    Args:
        logger_name (str): Name of the logger
        parent_logger (logging.Logger, optional): Parent logger to inherit timestamp from
        subfolder (str, optional): Subfolder to create within the timestamp directory

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create specific logger for this function
    if parent_logger:
        # Extract timestamp from parent logger's file handler
        timestamp = parent_logger.handlers[1].baseFilename.split("/")[-2]
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = f"exps/{timestamp}"
    if subfolder:
        log_dir = f"{log_dir}/{subfolder}"

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    child_logger = logging.getLogger(logger_name)
    child_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicate logs
    if child_logger.handlers:
        for handler in child_logger.handlers:
            child_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(f"{log_dir}/{logger_name}.log")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    child_logger.addHandler(console_handler)
    child_logger.addHandler(file_handler)

    child_logger.info(f"{logger_name} logger initialized - writing to {log_dir}/{logger_name}.log")

    return child_logger
