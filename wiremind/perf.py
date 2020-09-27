# -*- coding: utf-8 -*-
"""
The perf module.

Contain performance-related code.

Created by Romain Mondon-Cancel on 2020-09-26 21:45:44
"""

import functools as ft
import logging
import time
import typing as t


def timer(function: t.Callable) -> t.Callable:
    """
    Decorate a function to log the time spent executing it.

    Args:
        function: The function to decorate.

    Returns:
        The decorated function, logging the time spent when executing it.
    """

    @ft.wraps(function)
    def wrapped(*args: t.Any, **kwargs: t.Any) -> t.Any:
        start = time.perf_counter()
        ans = function(*args, **kwargs)
        logging.getLogger(function.__module__).debug(
            f"Computed the result of {function} in {time.perf_counter() - start:.05f} "
            "seconds."
        )
        return ans

    return wrapped
