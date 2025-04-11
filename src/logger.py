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
        self._timestamp = None
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

    def get_timestamp(self):
        """Get the timestamp used by this logger."""
        return self._timestamp

    def get_instance_id(self):
        """Get the instance ID used by this logger."""
        return self._instance_id

    def setup(
        self,
        parent_logger=None,
        debug=False,
        instance_id=None,
        root_dir="exps",
        print_to_stdout=False,
    ):
        """
        Set up the logger with file handler and optionally console handler.

        Args:
            parent_logger (logging.Logger, optional): Parent logger to inherit timestamp and debug flag from
            debug (bool, optional): Whether to add a debug tag to the log directory. If parent_logger is provided,
                                  this value will be overridden by the parent's debug status.
            instance_id (str, optional): Unique identifier for this logger instance. If parent_logger is provided,
                                      this value will be overridden by the parent's instance_id.
            root_dir (str, optional): Root directory for storing logs. Defaults to "exps".
            print_to_stdout (bool, optional): Whether to print logs to stdout. Defaults to False.

        Returns:
            CustomLogger: Self, for method chaining
        """
        # Determine if this is a root logger
        self._is_root = parent_logger is None
        if self._is_root:
            os.makedirs(root_dir, exist_ok=True)
            print(f"Root directory: {root_dir}")

        # Set parent logger reference
        self._parent_logger = parent_logger

        # Determine timestamp, debug status, and instance_id
        if parent_logger and isinstance(parent_logger, CustomLogger):
            # Inherit timestamp, debug status, and instance_id from parent
            self._timestamp = parent_logger.get_timestamp()
            self._debug = parent_logger.is_debug()
            self._instance_id = parent_logger.get_instance_id()
        else:
            # Create new timestamp and set instance_id
            self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._instance_id = instance_id
            if debug and self._is_root:
                self._debug = True

        # Create log directory
        if parent_logger is None:
            # Root logger always creates a new base directory
            log_dir = f"{root_dir}/{self._timestamp}@{self._instance_id}"
            if self._debug:
                log_dir = f"{log_dir}_debug"
        else:
            # Child logger uses parent's directory and adds its name as subfolder
            base_dir = parent_logger.get_logdir()
            log_dir = f"{base_dir}/{self.name}"

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
        if print_to_stdout:
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
    parent_logger=None,
    debug=False,
    instance_id=None,
    root_dir="exps",
    print_to_stdout=False,
):
    """
    Set up and return a logger that writes to file and optionally to console.

    Args:
        logger_name (str): Name of the logger (default: "main")
        parent_logger (logging.Logger, optional): Parent logger to inherit timestamp from
        debug (bool, optional): Whether to add a debug tag to the log directory
        instance_id (str, optional): Unique identifier for this logger instance
        root_dir (str, optional): Root directory for storing logs. Defaults to "exps".
        print_to_stdout (bool, optional): Whether to print logs to stdout. Defaults to False.

    Returns:
        CustomLogger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)

    if isinstance(logger, CustomLogger):
        return logger.setup(
            parent_logger=parent_logger,
            debug=debug,
            instance_id=instance_id,
            root_dir=root_dir,
            print_to_stdout=print_to_stdout,
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
    child_logger = setup_logger(logger_name="child", parent_logger=main_logger)
    child_logdir = child_logger.get_logdir()
    print(f"Child logger directory: {child_logdir}")

    # Use the log directory in your application
    main_logger.info(f"Using log directory: {main_logger.get_logdir()}")

    # You can use the log directory to save additional files
    output_file = f"{child_logger.get_logdir()}/output.json"
    print(f"Additional files can be saved to: {output_file}")
