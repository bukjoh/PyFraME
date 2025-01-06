import logging
from collections import defaultdict
from typing import List, Optional, DefaultDict


class CustomLogHandler(logging.Handler):
    """
    A custom log handler that stores log messages in a dictionary categorized by log levels.
    """

    def __init__(self) -> None:
        """
        Initialize the CustomLogHandler.
        """
        super().__init__()
        self.log_messages: DefaultDict[str, List[str]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        Args:
            record (logging.LogRecord): The log record to be logged.
        """
        log_entry = self.format(record)
        self.log_messages[record.levelname].append(log_entry)


class LogManager:
    """
    A singleton log manager that handles logging configuration and log message retrieval.
    """
    _instance: Optional['LogManager'] = None

    def __new__(cls, level: Optional[int] = logging.DEBUG, log_to_console: bool = True) -> 'LogManager':
        """
        Create or return the singleton instance of LogManager.

        Args:
            level (Optional[int]): The logging level.
            log_to_console (bool): Flag to determine if logs should also be output to the console.

        Returns:
            LogManager: The singleton instance of LogManager.
        """
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance.logger = logging.getLogger('PyFraME')
            cls._instance.handler = CustomLogHandler()
            cls._instance.console_handler = logging.StreamHandler() if log_to_console else None

            cls._instance._set_handlers_level(level)
            cls._instance._set_formatters(level)

            cls._instance.logger.addHandler(cls._instance.handler)
            if cls._instance.console_handler:
                cls._instance.logger.addHandler(cls._instance.console_handler)
        return cls._instance

    def _set_handlers_level(self, level: int) -> None:
        """
        Set the logging level for all handlers.

        Args:
            level (int): The logging level to be set.
        """
        self.logger.setLevel(level)
        self.handler.setLevel(level)
        if self.console_handler:
            self.console_handler.setLevel(level)

    def _set_formatters(self, level: int) -> None:
        """
        Set the formatters for the handlers based on the logging level.

        Args:
            level (int): The logging level to determine the formatter.
        """
        if level == logging.DEBUG:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
        else:
            formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        if self.console_handler:
            self.console_handler.setFormatter(formatter)

    def set_level(self, level: int) -> None:
        """
        Set the logging level.

        Args:
            level (int): The logging level to be set.
        """
        self._set_handlers_level(level)
        self._set_formatters(level)

    def get_logs(self) -> str:
        """
        Retrieve all log messages that are at or above the current logging level.

        Returns:
            str: The concatenated log messages.
        """
        level = self.logger.level
        all_messages: List[str] = []
        for lvl in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            if logging.getLevelName(lvl) >= level:
                all_messages.extend(self.handler.log_messages[lvl])
        return "\n".join(all_messages)

    def reset(self) -> None:
        """
        Reset the log manager by removing all handlers and clearing stored log messages.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.handler = CustomLogHandler()
        self.console_handler = logging.StreamHandler() if self.console_handler else None

        self._set_formatters(self.logger.level)
        self._set_handlers_level(self.logger.level)

        self.logger.addHandler(self.handler)
        if self.console_handler:
            self.logger.addHandler(self.console_handler)

        self.handler.log_messages.clear()

    @property
    def instance(self) -> 'LogManager':
        """
        Property to get the singleton instance.

        Returns:
            LogManager: The singleton instance of LogManager.
        """
        return self._instance


# Create a singleton instance with a specific logging level
log_manager = LogManager(logging.INFO, log_to_console=False)
