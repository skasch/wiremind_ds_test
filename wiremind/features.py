# -*- coding: utf-8 -*-
"""
The features module.

Extract useful features for training the model.

Created by Romain Mondon-Cancel on 2020-09-25 22:26:54
"""

import bisect
import functools as ft
import logging
import random
import typing as t

import pandas as pd
import sklearn.model_selection

from . import perf

logger = logging.getLogger(__name__)


def add_trip_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trip id to ``df``, identifying a unique trip booking.

    Args:
        df: The dataframe to update.

    Returns:
        A dataframe containing a string column identifying a unique trip booking,
        represented by the time of departure to the minute and the origin and
        destination stations.
    """
    return df.assign(
        trip_id=lambda df: [
            f"{datetime.strftime('%Y%m%d%H%M')}{start}{end}"
            for datetime, start, end in zip(
                df.departure_datetime, df.station_origin, df.station_destination
            )
        ]
    )


def add_trip_path(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trip path to ``df``, identifying the path of a trip.

    Args:
        df: The dataframe to update.

    Returns:
        A dataframe containing a string column representing the path of each trip,
        represented by the name of the origin and the destination stations.
    """
    return df.assign(
        trip_path=lambda df: [
            f"{start}{end}"
            for start, end in zip(df.station_origin, df.station_destination)
        ]
    )


def add_sale_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sale time delta data to ``df``.

    Args:
        df: The dataframe to update.

    Returns:
        A dataframe containing two additional columns, ``sale_timedelta`` containing
        the time difference between the departure and the sale time, and ``sale_day``
        containing the number of days between the sale and the departure.
        ``sale_day=0`` means the sale happened after the departure.
    """
    # fmt: off
    return (
        df
        .assign(sale_timedelta=lambda df: df.sale_datetime - df.departure_datetime)
        .assign(sale_day=lambda df: df.sale_timedelta.dt.days)
    )
    # fmt: on


def add_cancel_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cancellation time delta data to ``df``.

    Args:
        df: The dataframe to update.

    Returns:
        A dataframe containing two additional columns, ``cancel_timedelta`` containing
        the time difference between the departure and the cancellation time if it
        exists, and ``cancel_day`` containing the number of days between the
        cancellation and the departure. ``cancel_day=0`` means the cancellation
        happened after the departure.
    """
    # fmt: off
    return (
        df
        .assign(cancel_timedelta=lambda df: df.cancel_datetime - df.departure_datetime)
        .assign(cancel_day=lambda df: df.cancel_timedelta.dt.days)
    )
    # fmt: on


@perf.timer
def one_hot_encode(
    df: pd.DataFrame,
    values: t.List,
    columns: t.List[str],
    function: t.Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """
    One-hot encode some values.

    Args:
        df: The dataframe to update
        values: The expected different values to one-hot encode.
        columns: The names of the columns corresponding to each one-hot encoded value.
        function: The function applied to ``df``, returning a series containing the
            values to one-hot encode.

    Returns:
        A dataframe containing additional columns, one for each one-hot encoded value,
        containing ``1`` if ``function(df)`` is equal to the corresponding value and
        ``0`` otherwise.
    """
    series = function(df)
    return df.assign(
        **{
            columns[idx]: (series == value).astype(int)
            for idx, value in enumerate(values)
        }
    )


HOURS = list(range(24))
HOUR_COLUMNS = [f"hour_{h}" for h in HOURS]

add_hour = ft.partial(
    one_hot_encode,
    values=HOURS,
    columns=HOUR_COLUMNS,
    function=lambda df: df.departure_datetime.dt.hour,
)
"""
Add information about the hour of the departure time of the train.

Args:
    df: The dataframe to update.

Returns:
    A dataframe containing additional, one-hot encoded columns for each hour of the
    day, with 1 if the train departs on that hour.
"""


DAYS_OF_WEEK = list(range(7))
DAY_OF_WEEK_COLUMNS = [f"day_of_week_{d}" for d in DAYS_OF_WEEK]

add_day_of_week = ft.partial(
    one_hot_encode,
    values=DAYS_OF_WEEK,
    columns=DAY_OF_WEEK_COLUMNS,
    function=lambda df: df.departure_datetime.dt.weekday,
)
"""
Add information about the day of week of the departure date of the train.

Args:
    df: The dataframe to update.

Returns:
    A dataframe containing additional, one-hot encoded columns for each day of the
    week, with 1 if the train departs on that day.
"""


PERIODS = list(range(24))
PERIOD_COLUMNS = [f"period_{p}" for p in PERIODS]

add_period = ft.partial(
    one_hot_encode,
    values=PERIODS,
    columns=PERIOD_COLUMNS,
    function=lambda df: (
        (df.departure_datetime.dt.month - 1) * 2 + (df.departure_datetime.dt.day > 15)
    ),
)
"""
Add information about the period in the year of the departure date of the train.

Args:
    df: The dataframe to update.

Returns:
    A dataframe containing additional, one-hot encoded columns for each period of
    the year, with 1 if the train departs on that period. A period is roughly
    defined as a half-month. The first period of each month ends on the 15th day
    of that month, and the second period starts on the 16th day.
"""


PATHS = ["AB", "BA"]
PATH_COLUMNS = [f"path_{p}" for p in PATHS]

add_path = ft.partial(
    one_hot_encode,
    values=PATHS,
    columns=PATH_COLUMNS,
    function=lambda df: df.trip_path,
)
"""
Add information about the path of the trip.

Args:
    df: The dataframe to update.

Returns:
    A dataframe containing additional, one-hot encoded columns for each unique path
    the train can take; in that case, the trip can only travel between two stations A
    and B, which means the two unique paths are AB and BA.
"""


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all the useful features to enrich a dataframe.

    Args:
        df: The dataframe to update.

    Returns:
        A dataframe containing additional, useful columns.
    """
    return (
        df.pipe(add_trip_id)
        .pipe(add_trip_path)
        .pipe(add_sale_delta)
        .pipe(add_cancel_delta)
        .pipe(add_hour)
        .pipe(add_day_of_week)
        .pipe(add_period)
        .pipe(add_path)
    )


INFO_COLUMNS = HOUR_COLUMNS + DAY_OF_WEEK_COLUMNS + PERIOD_COLUMNS + PATH_COLUMNS

DEFAULT_TEST_SIZE = 0.1


def train_test_split(
    df: pd.DataFrame,
    train_size: t.Optional[float] = None,
    test_size: t.Optional[float] = None,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe in random groups along the ``trip_id`` column.

    Args:
        df: The dataframe to split. The dataframe must contain the column ``trip_id``.
        train_size: Defaults to ``None``. The size of the training set, as a ratio. Must
            be between 0 and 1. If ``None``, set such that
            ``train_size + test_size = 1``.
        test_size: Defaults to ``None``. The size of the test set, as a ratio. If both
            ``train_size`` and ``test_size`` are ``None``, defaults to 0.1. Must be
            between 0 and 1. If ``None``, set such that ``train_size + test_size = 1``.

    Returns:
        Two datasets extracted from ``df``. The first one contains ``train_size`` of the
        unique trip ids of ``df``, while the second one contains ``test_size`` of them.
    """
    if train_size is None and test_size is None:
        test_size = DEFAULT_TEST_SIZE
    train_ids, test_ids = sklearn.model_selection.train_test_split(
        df.trip_id.unique(), train_size=train_size, test_size=test_size
    )
    return (
        df.loc[lambda df: df.trip_id.isin(train_ids)],
        df.loc[lambda df: df.trip_id.isin(test_ids)],
    )


def split_X_y(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe ``df`` into ``X`` and ``y`` parts.

    Args:
        df: The dataframe to split. Must contain the column ``demand``, the target.

    Returns:
        A dataframe containing the input ``X`` for the model, consisting of all the
        columns except for ``demand``, and the target ``y`` for the model, the column
        ``demand``.
    """
    X_cols = [c for c in df.columns if c != "demand"]
    return df[X_cols], df.demand


PriceGenerator = t.Callable[[float, float, int], t.List[float]]


def generate_prices(min_price: float, max_price: float, n_prices: int) -> t.List[float]:
    """
    Generate a sorted list of prices in a given range.

    Args:
        min_price: The minimum price of tickets.
        max_price: The maximum price of tickets.
        n_prices: The number of prices to generate.

    Returns:
        A list of prices uniformly picked over a range [m, M] such that 5% of values
        are picked between m and min_price, 90% of values between min_price and
        max_price, and 5% between max_price and M.
    """
    delta = (max_price - min_price) / 0.9
    adjusted_min = min_price - 0.05 * delta
    return sorted(
        round(adjusted_min + delta * random.random(), 2) for _ in range(n_prices)
    )


def compute_demands(
    sale_prices: t.List[float], target_prices: t.List[float]
) -> t.List[int]:
    """
    Compute the demand for each of target prices, given the full list of sale prices.

    Using binary search, this algorithm is in O(n log m), where n is the number of
    target prices and m is the number of sale prices.

    Args:
        sale_prices: The sorted list of prices at which tickets were sold.
        target_prices: The prices to compute the demand for.

    Returns:
        The list of demands for each target price. The demand is the number of tickets
        sold for at least the target value.
    """
    n = len(sale_prices)
    sale_idx = 0
    ans = []
    for price in target_prices:
        sale_idx = bisect.bisect_left(sale_prices, price, lo=sale_idx)
        ans.append(n - sale_idx)
    return ans


def merge_sorted(arr1: t.List[float], arr2: t.List[float]) -> t.List[float]:
    """
    Merge two sorted arrays into a third sorted array.

    Using two pointers, this algorithm is in O(n1 + n2), where n1 and n2 are
    respectively the lengths of ``arr1`` and ``arr2``.

    Args:
        arr1: The first array to merge.
        arr2: The second array to merge.

    Returns:
        A sorted array containing all values from ``arr1`` and ``arr2``.
    """
    ans = []
    idx1 = 0
    idx2 = 0
    n1 = len(arr1)
    n2 = len(arr2)
    while idx1 < n1 or idx2 < n2:
        if idx1 == n1:
            ans.extend(arr2[idx2:])
            idx2 = n2
        elif idx2 == n2:
            ans.extend(arr1[idx1:])
            idx1 = n1
        elif arr1[idx1] < arr2[idx2]:
            ans.append(arr1[idx1])
            idx1 += 1
        else:
            ans.append(arr2[idx2])
            idx2 += 1
    return ans


def describe_previous_demand(
    previous_total_demand: int,
    previous_prices: t.List[float],
    day: int,
    prices: t.List[float],
) -> pd.DataFrame:
    """
    Describe the demand prior to a given day for a trip given a list of prices.

    Args:
        previous_total_demand: The total number of tickets sold up to the day before.
        previous_prices: The list of all prices for tickets sold up to the day before.
        day: The day considered, relative to the departure day.
        prices: The list of prices to compute demand for.

    Returns:
        A dataframe containing, for each price in ``prices``, the corresponding demand
        on all prior days, the total demand on all prior days, the prices, and the day
        number itself.
    """
    previous_price_demands = compute_demands(previous_prices, prices)
    return pd.DataFrame(
        dict(
            day=day,
            price=prices,
            previous_total_demand=previous_total_demand,
            previous_price_demand=previous_price_demands,
        )
    )


def describe_demand(
    day_prices: t.List[float],
    previous_total_demand: int,
    previous_prices: t.List[float],
    day: int,
    prices: t.List[float],
) -> pd.DataFrame:
    """
    Describe the demand on a given day for a trip given a list of prices.

    Args:
        day_prices: The prices at which the tickets were sold on a given day.
        previous_total_demand: The total number of tickets sold up to the day before.
        previous_prices: The list of all prices for tickets sold up to the day before.
        day: The day considered, relative to the departure day.
        prices: The list of prices to compute demand for.

    Returns:
        A dataframe containing, for each price in ``prices``, the corresponding demand
        on that day, as well as the corresponding demand on all prior days, the total
        demand on all prior days, the prices, and the day number itself.
    """
    price_demands = compute_demands(day_prices, prices)
    return describe_previous_demand(
        previous_total_demand, previous_prices, day, prices
    ).assign(demand=price_demands)


DEFAULT_N_PRICES = 20


def process_trip_at_day(
    trip_df: pd.DataFrame, day: int, prices: t.List[float]
) -> pd.DataFrame:
    """
    Process the trip data on a given day; does not include the demand on that day.

    Args:
        trip_df: The dataframe of the data for a given trip.
        day: The day to consider.
        prices: The list of prices to process the data for.

    Returns:
        A dataframe containing all the contextual information required to describe what
        is known of a trip prior ot ``day`` for a given list of ``prices``. This
        includes one-hot encoded information about the trip itself, the hour of the
        departure, the day of the week, and the period during the year when the trip
        occurred, as well as information about bookings prior to ``day``, as described
        by ``describe_previous_demand``.
    """
    trip_df = trip_df.sort_values("price")
    trip_info = trip_df[INFO_COLUMNS].iloc[0]
    previous_df = trip_df.loc[lambda df: df.sale_day < day]
    previous_total_demand = previous_df.shape[0]
    previous_prices = previous_df.price.tolist()
    return describe_previous_demand(
        previous_total_demand, previous_prices, day, prices
    ).assign(**trip_info)


@perf.timer
def process_trip(
    trip_df: pd.DataFrame,
    n_prices: int = DEFAULT_N_PRICES,
) -> pd.DataFrame:
    """
    Process a trip dataframe to return the data as expected by the model.

    Args:
        trip_df: The dataframe corresponding to a unique trip.
        n_prices: Defaults to ``DEFAULT_N_PRICES``. The number of prices to generate.

    Returns:
        A dataframe containing, for each day when sales occurred, ``n_prices`` lines,
        where the script generates that many random prices. The features are built by
        the ``describe_demand`` function.
    """
    rows = []
    previous_total_demand = 0
    previous_prices: t.List[float] = []
    min_price = trip_df.price.min()
    max_price = trip_df.price.max()
    trip_info = trip_df[INFO_COLUMNS].iloc[0]
    for day, day_trip_df in trip_df.sort_values("price").groupby("sale_day"):
        prices = generate_prices(min_price, max_price, n_prices)
        day_prices = day_trip_df.price.tolist()
        day_demand = describe_demand(
            day_prices,
            previous_total_demand,
            previous_prices,
            day,
            prices,
        )
        rows.append(day_demand)
        previous_total_demand += day_trip_df.shape[0]
        previous_prices = merge_sorted(previous_prices, day_prices)
    return pd.concat(rows).assign(**trip_info)


def process_df(df: pd.DataFrame, n_prices: int = DEFAULT_N_PRICES) -> pd.DataFrame:
    """
    Process all trips in ``df`` to generate a full dataframe as expected by the model.

    Args:
        df: The dataframe to process.
        n_prices: Defaults to ``DEFAULT_N_PRICES``. The number of prices to generate.

    Returns:
        A full dataframe, as processed by ``process_trip`` for each unique trip in
        ``df``.
    """
    groups = []
    for trip_id, trip_df in df.groupby("trip_id"):
        groups.append(process_trip(trip_df, n_prices))
        logger.info(f"Processed trip {trip_id} successfully.")
    return pd.concat(groups)
