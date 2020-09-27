# -*- coding: utf-8 -*-
"""
The data module.

Read the data.

Created by Romain Mondon-Cancel on 2020-09-25 21:49:03
"""

import pandas as pd

SCHEMA = {
    "departure_datetime": str,
    "station_origin": str,
    "station_destination": str,
    "sale_datetime": str,
    "cancel_datetime": str,
    "price": float,
}

DATETIME_COLUMNS = ["departure_datetime", "sale_datetime", "cancel_datetime"]


def read_csv(filepath: str) -> pd.DataFrame:
    """
    Read the data contained in the file at ``filepath``.

    Args:
        filepath: The path to a ``CSV`` file containing the data.

    Returns:
        The data contained in ``filepath``.
    """
    return pd.read_csv(filepath, dtype=SCHEMA, parse_dates=DATETIME_COLUMNS)
