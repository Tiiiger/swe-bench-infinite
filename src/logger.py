import datetime
import logging
import os
import pathlib


class CustomLogger(logging.Logger):
    """Custom logger class that extends the standard Logger with additional functionality."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.log_dir = None
        self._debug = False
        self._is_root = False
        self._parent_logger = None
        self._instance_id = None

    def get_logdir(self) -> str:
        """Get the log directory used by this logger."""
        if self.log_dir is None:
            raise ValueError("Log directory not set")
        return self.log_dir

    def is_debug(self):
        """Get whether this logger is in debug mode."""
        return self._debug

    def is_root(self):
        """Get whether this logger is a root logger."""
        return self._is_root

    def get_parent(self):
        """Get the parent logger if this is a child logger."""
        return self._parent_logger

    def get_instance_id(self):
        """Get the instance ID used by this logger."""
        return self._instance_id

    def setup(
        self,
        logger_name,
        parent_logger=None,
        debug=False,
        instance_id=None,
        root_dir="exps",
    ):
        """
        Set up the logger with file handler and optionally console handler.

        Args:
            parent_logger (logging.Logger, optional): Parent logger to inherit debug flag from
            debug (bool, optional): Whether to add a debug tag to the log directory. If parent_logger is provided,
                                  this value will be overridden by the parent's debug status.
            instance_id (str): Unique identifier for this logger instance. If parent_logger is provided,
                             this value will be overridden by the parent's instance_id.
            root_dir (str, optional): Root directory for storing logs. Defaults to "exps".

        Returns:
            CustomLogger: Self, for method chaining
        """
        # Verify instance_id is provided for root loggers
        if parent_logger is None and instance_id is None:
            raise ValueError("instance_id is required for root loggers")

        # Determine if this is a root logger
        self._is_root = parent_logger is None
        if self._is_root:
            os.makedirs(root_dir, exist_ok=True)

        # Set parent logger reference
        self._parent_logger = parent_logger

        # Determine debug status and instance_id
        if parent_logger and isinstance(parent_logger, CustomLogger):
            # Inherit debug status and instance_id from parent
            self._debug = parent_logger.is_debug()
            self._instance_id = parent_logger.get_instance_id()
        else:
            # Set instance_id and debug flag
            self._instance_id = instance_id
            if debug and self._is_root:
                self._debug = True

        # Create log directory
        if parent_logger is None:
            # Root logger creates a directory with just the instance_id
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"{root_dir}/{timestamp}@{self._instance_id}"
            if self._debug:
                log_dir = f"{log_dir}_debug"
        else:
            # Child logger uses parent's directory and adds its name as subfolder
            base_dir = parent_logger.get_logdir()
            log_dir = f"{base_dir}/{logger_name}"

        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Configure log file
        log_file = f"{log_dir}/{self.name}.log"

        # Remove any existing handlers to avoid duplicate logs
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set log level and prevent propagation
        self.setLevel(logging.DEBUG)
        self.propagate = False  # Prevent messages from being passed to parent loggers

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # Add console handler if requested
        if debug:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

        # Log initialization message
        if self._is_root:
            self.info(f"Root logger initialized - logging to {log_file}")
        else:
            self.info(f"{self.name} logger initialized - writing to {log_file}")

        return self


# Register our custom logger class
logging.setLoggerClass(CustomLogger)


def setup_logger(
    logger_name="main",
    instance_id=None,
    parent_logger=None,
    debug=False,
    root_dir="exps",
):
    """
    Set up and return a logger that writes to file and optionally to console.

    Args:
        logger_name (str): Name of the logger (default: "main")
        parent_logger (logging.Logger, optional): Parent logger to inherit from
        debug (bool, optional): Whether to add a debug tag to the log directory
        instance_id (str): Unique identifier for this logger instance (required if parent_logger is None)
        root_dir (str, optional): Root directory for storing logs. Defaults to "exps".

    Returns:
        CustomLogger: Configured logger instance
    """
    # Inherit instance_id from parent_logger if provided and instance_id is a CustomLogger
    if parent_logger is not None and isinstance(parent_logger, CustomLogger):
        inherited_instance_id = parent_logger.get_instance_id()
        if instance_id is None:
            instance_id = inherited_instance_id

    # Ensure instance_id is not None when parent_logger is None
    if parent_logger is None and instance_id is None:
        raise ValueError("instance_id is required when parent_logger is None")

    logger = logging.getLogger(f"{logger_name}_{instance_id}")

    if isinstance(logger, CustomLogger):
        return logger.setup(
            logger_name=logger_name,
            parent_logger=parent_logger,
            debug=debug,
            instance_id=instance_id,
            root_dir=root_dir,
        )
    else:
        # Fallback for non-CustomLogger instances (should not happen if setLoggerClass is working)
        raise TypeError(
            "Logger is not a CustomLogger instance. Make sure logging.setLoggerClass(CustomLogger) is called."
        )


# Example usage
if __name__ == "__main__":
    # Set up a main logger
    main_logger = setup_logger(debug=True, instance_id="test_run")
    main_logdir = main_logger.get_logdir()
    print(f"Main logger directory: {main_logdir}")

    # Set up a child logger
    child_logger = setup_logger(logger_name="child", parent_logger=main_logger, instance_id="1")
    child_logdir = child_logger.get_logdir()
    print(f"Child logger directory: {child_logdir}")

    # Use the log directory in your application
    main_logger.info(f"Using log directory: {main_logger.get_logdir()}")

    # You can use the log directory to save additional files
    output_file = f"{child_logger.get_logdir()}/output.json"
    print(f"Additional files can be saved to: {output_file}")
