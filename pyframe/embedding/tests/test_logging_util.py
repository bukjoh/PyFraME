import logging
import pytest
from pyframe.embedding.logging_util import LogManager


@pytest.fixture
def log_manager():
    # Ensure we start with a fresh LogManager instance for each test
    if LogManager._instance is not None:
        LogManager._instance = None
    return LogManager(logging.INFO)


def test_initial_logging_level(log_manager):
    assert log_manager.logger.level == logging.INFO
    assert log_manager.console_handler.level == logging.INFO


def test_logging_info_level(log_manager):
    log_manager.logger.info("Info level log")
    log_messages = log_manager.get_logs()
    assert "Info level log" in log_messages


def test_logging_debug_level(log_manager):
    log_manager.set_level(logging.DEBUG)
    log_manager.logger.debug("Debug level log")
    log_messages = log_manager.get_logs()
    assert "Debug level log" in log_messages


def test_logging_warning_level(log_manager):
    log_manager.logger.warning("Warning level log")
    log_messages = log_manager.get_logs()
    assert "Warning level log" in log_messages


def test_logging_error_level(log_manager):
    log_manager.logger.error("Error level log")
    log_messages = log_manager.get_logs()
    assert "Error level log" in log_messages


def test_logging_critical_level(log_manager):
    log_manager.logger.critical("Critical level log")
    log_messages = log_manager.get_logs()
    assert "Critical level log" in log_messages


def test_change_logging_level_to_error(log_manager):
    # Change logging level to ERROR
    log_manager.set_level(logging.ERROR)
    assert log_manager.logger.level == logging.ERROR
    assert log_manager.console_handler.level == logging.ERROR

    # Log messages at different levels
    log_manager.logger.debug("Debug level log")
    log_manager.logger.info("Info level log")
    log_manager.logger.warning("Warning level log")
    log_manager.logger.error("Error level log")
    log_manager.logger.critical("Critical level log")

    # Retrieve logs
    log_messages = log_manager.get_logs()

    # Verify that only ERROR and CRITICAL logs are shown
    assert "Debug level log" not in log_messages
    assert "Info level log" not in log_messages
    assert "Warning level log" not in log_messages
    assert "Error level log" in log_messages
    assert "Critical level log" in log_messages


def test_change_logging_level_to_debug(log_manager):
    # Change logging level to DEBUG
    log_manager.set_level(logging.DEBUG)
    assert log_manager.logger.level == logging.DEBUG
    assert log_manager.console_handler.level == logging.DEBUG

    # Log messages at different levels
    log_manager.logger.debug("Debug level log")
    log_manager.logger.info("Info level log")
    log_manager.logger.warning("Warning level log")
    log_manager.logger.error("Error level log")
    log_manager.logger.critical("Critical level log")

    # Retrieve logs
    log_messages = log_manager.get_logs()

    # Verify that all logs are shown
    assert "Debug level log" in log_messages
    assert "Info level log" in log_messages
    assert "Warning level log" in log_messages
    assert "Error level log" in log_messages
    assert "Critical level log" in log_messages


def test_change_logging_level_to_info(log_manager):
    # Change logging level to INFO
    log_manager.set_level(logging.INFO)
    assert log_manager.logger.level == logging.INFO
    assert log_manager.console_handler.level == logging.INFO

    # Log messages at different levels
    log_manager.logger.debug("Debug level log")
    log_manager.logger.info("Info level log")
    log_manager.logger.warning("Warning level log")
    log_manager.logger.error("Error level log")
    log_manager.logger.critical("Critical level log")

    # Retrieve logs
    log_messages = log_manager.get_logs()

    # Verify that INFO and higher logs are shown
    assert "Debug level log" not in log_messages
    assert "Info level log" in log_messages
    assert "Warning level log" in log_messages
    assert "Error level log" in log_messages
    assert "Critical level log" in log_messages


def test_change_logging_level_to_warning(log_manager):
    # Change logging level to WARNING
    log_manager.set_level(logging.WARNING)
    assert log_manager.logger.level == logging.WARNING
    assert log_manager.console_handler.level == logging.WARNING

    # Log messages at different levels
    log_manager.logger.debug("Debug level log")
    log_manager.logger.info("Info level log")
    log_manager.logger.warning("Warning level log")
    log_manager.logger.error("Error level log")
    log_manager.logger.critical("Critical level log")

    # Retrieve logs
    log_messages = log_manager.get_logs()

    # Verify that WARNING and higher logs are shown
    assert "Debug level log" not in log_messages
    assert "Info level log" not in log_messages
    assert "Warning level log" in log_messages
    assert "Error level log" in log_messages
    assert "Critical level log" in log_messages
