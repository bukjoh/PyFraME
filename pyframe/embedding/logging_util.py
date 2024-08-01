import logging
from collections import defaultdict
from typing import List, Optional, DefaultDict


class CustomLogHandler(logging.Handler):
    """
    A custom logging handler that collects log messages in a dictionary
    categorized by log level.

    Attributes:
        log_messages (DefaultDict[str, List[str]]): A dictionary storing log messages
            categorized by their log levels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_messages: DefaultDict[str, List[str]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        This method formats the log record and stores it in the log_messages
        dictionary categorized by the log level.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        log_entry = self.format(record)
        self.log_messages[record.levelname].append(log_entry)


class LogManager:
    """
    A class to manage logging configuration and retrieval of log messages.
    """

    _instance = None

    def __new__(cls, level: Optional[int] = logging.DEBUG) -> 'LogManager':
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance.logger = logging.getLogger('PyFraME')
            cls._instance.handler = CustomLogHandler()
            cls._instance.console_handler = logging.StreamHandler()

            cls._instance._set_handlers_level(level)  # Set level for logger and handlers
            cls._instance._set_formatters(level)      # Set formatters based on the level

            cls._instance.logger.addHandler(cls._instance.handler)
            cls._instance.logger.addHandler(cls._instance.console_handler)
        return cls._instance

    def _set_handlers_level(self, level: int) -> None:
        """
        Set the logging level for the logger and all its handlers.

        Args:
            level (int): The new logging level.
        """
        self.logger.setLevel(level)
        self.handler.setLevel(level)
        self.console_handler.setLevel(level)

    def _set_formatters(self, level: int) -> None:
        """
        Set formatters for the handlers based on the logging level.

        Args:
            level (int): The current logging level.
        """
        if level == logging.DEBUG:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
        else:
            formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

    def set_level(self, level: int) -> None:
        """
        Set the logging level for the logger and its handlers.

        Args:
            level (int): The new logging level.
        """
        self._set_handlers_level(level)
        self._set_formatters(level)  # Update formatters when level changes

    def get_logs(self) -> str:
        """
        Retrieve collected log messages filtered by the set logging level.

        Returns:
            str: A string of the collected log messages.
        """
        level = self.logger.level
        all_messages: List[str] = []
        for lvl in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            if logging.getLevelName(lvl) >= level:
                all_messages.extend(self.handler.log_messages[lvl])
        return "\n".join(all_messages)

    def reset(self) -> None:
        """
        Reset the logger by clearing all saved messages and reinitializing the logger.
        """
        # Clear all handlers from the logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Reinitialize the handler and console handler
        self.handler = CustomLogHandler()
        self.console_handler = logging.StreamHandler()

        # Set formatters and level
        self._set_formatters(self.logger.level)
        self._set_handlers_level(self.logger.level)

        self.logger.addHandler(self.handler)
        self.logger.addHandler(self.console_handler)

        # Clear log messages
        self.handler.log_messages.clear()

    @property
    def instance(self):
        return self._instance


# Create a singleton instance with a specific logging level
log_manager = LogManager(logging.INFO)
