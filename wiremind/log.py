# -*- coding: utf-8 -*-
"""
The log module.

Handle logging facilities.

Created by Romain Mondon-Cancel on 2020-09-26 20:57:24
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s [%(levelname)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Set up the logger for the module ``name``.

    Args:
        name: The name of the module to set up the logger for.

    Returns:
        The set up logger for the module ``name``.
    """
    logger = logging.getLogger(name)
    logging_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    logging_handler = logging.StreamHandler(sys.stderr)
    logging_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_handler)
    return logger
