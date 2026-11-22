"""Functions for handling logging more easily."""

from typing import Any
import logging
from src.config.config_utils import Config


def get_logger(log_level: str = None) -> Any:
    """
    Function to create a logger object.
    Args:
        log_level (string): The level of logging to use. Default is INFO.
    Returns:
        logger (any): The logger object
    """

    config_model = Config("../src/config/config_model.yml")

    if log_level is None:
        try:
            log_level = config_model.get_config_value("logger_config.level")
        except KeyError:
            log_level = "INFO"

    logger = logging.getLogger(__name__)

    # Handle where the logs go: stdout/file?
    logger.handlers = []  # Clean handlers list
    handler = logging.StreamHandler()

    # Format of our logs
    msg_fmt = "%(asctime)s - %(levelname)s - %(filename)s - [%(funcName)s: %(lineno)d] - %(message)s"
    time_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(msg_fmt, time_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger
