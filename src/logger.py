import datetime
import logging
import pathlib


class CustomLogger(logging.Logger):
    """Custom logger class that extends the standard Logger with additional functionality."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.log_dir = None

    def get_logdir(self):
        """Get the log directory used by this logger."""
        return self.log_dir

    def setup(self, parent_logger=None, subfolder="", debug=False):
        """
        Set up the logger with console and file handlers.

        Args:
            parent_logger (logging.Logger, optional): Parent logger to inherit timestamp from
            subfolder (str, optional): Subfolder to create within the timestamp directory
            debug (bool, optional): Whether to add a debug tag to the log directory

        Returns:
            CustomLogger: Self, for method chaining
        """
        # Determine timestamp from parent logger or create new one
        if parent_logger and hasattr(parent_logger, "handlers") and len(parent_logger.handlers) > 1:
            timestamp = parent_logger.handlers[1].baseFilename.split("/")[-2]
            # Remove debug suffix if present
            if timestamp.endswith("_debug"):
                timestamp = timestamp[:-6]
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        log_dir = f"exps/{timestamp}"
        if debug and self.name == "main":
            log_dir = f"{log_dir}_debug"
        if subfolder:
            log_dir = f"{log_dir}/{subfolder}"

        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Configure log file
        log_file = f"{log_dir}/{self.name}.log"

        # Remove any existing handlers to avoid duplicate logs
        if self.handlers:
            for handler in self.handlers:
                self.removeHandler(handler)

        # Set log level
        self.setLevel(logging.DEBUG)

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
        self.addHandler(console_handler)
        self.addHandler(file_handler)

        # Log initialization message
        if self.name == "main":
            self.info(f"Logging to {log_file}")
        else:
            self.info(f"{self.name} logger initialized - writing to {log_file}")

        return self


# Register our custom logger class
logging.setLoggerClass(CustomLogger)


def setup_logger(logger_name="main", parent_logger=None, subfolder="", debug=False):
    """
    Set up and return a logger that writes to both console and a file.

    Args:
        logger_name (str): Name of the logger (default: "main")
        parent_logger (logging.Logger, optional): Parent logger to inherit timestamp from
        subfolder (str, optional): Subfolder to create within the timestamp directory
        debug (bool, optional): Whether to add a debug tag to the log directory

    Returns:
        CustomLogger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)

    if isinstance(logger, CustomLogger):
        return logger.setup(parent_logger=parent_logger, subfolder=subfolder, debug=debug)
    else:
        # Fallback for non-CustomLogger instances (should not happen if setLoggerClass is working)
        raise TypeError(
            "Logger is not a CustomLogger instance. Make sure logging.setLoggerClass(CustomLogger) is called."
        )


# Example usage
if __name__ == "__main__":
    # Set up a main logger
    main_logger = setup_logger(debug=True)
    main_logdir = main_logger.get_logdir()
    print(f"Main logger directory: {main_logdir}")

    # Set up a child logger
    child_logger = setup_logger(logger_name="child", parent_logger=main_logger, subfolder="child")
    child_logdir = child_logger.get_logdir()
    print(f"Child logger directory: {child_logdir}")

    # Use the log directory in your application
    main_logger.info(f"Using log directory: {main_logger.get_logdir()}")

    # You can use the log directory to save additional files
    output_file = f"{child_logger.get_logdir()}/output.json"
    print(f"Additional files can be saved to: {output_file}")
