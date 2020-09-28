# -*- coding: utf-8 -*-
"""
The model module.

Describe the model.

Created by Romain Mondon-Cancel on 2020-09-25 22:55:56
"""

import logging
import os
import typing as t

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features

logger = logging.getLogger(__name__)


class DemandModel:

    """
    Model the demand for trips.

    Args:
        xgb_path: Defaults to ``os.path.join("model", "xgb_model")``. The path where
            the trained model should be saved to and loaded from.
        xgb_args: Defaults to ``None``. The additional arguments to pass to the
            ``XGBRegressor`` model instance. If ``None``, is set to ``{}``.
    """

    def __init__(
        self,
        xgb_path: str = os.path.join("model", "xgb_model"),
        xgb_args: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        if xgb_args is None:
            xgb_args = {}
        self.xgb_path = xgb_path
        self.trip_df: t.Optional[pd.DataFrame] = None
        self.initialize_model(**xgb_args)

    def initialize_model(self, **xgb_args: t.Any) -> None:
        """
        Initialize the model.

        If the model already exists at ``self.xgb_path``, loads the trained model from
        that location.

        Args:
            xgb_args: Additional arguments passed to the ``XGBRegressor`` model
                instance.
        """
        logger.info("Initializing model.")
        self.model = xgb.XGBRegressor(**xgb_args)
        if os.path.exists(self.xgb_path):
            logger.info(f"Found existing model at {self.xgb_path}; loading model.")
            self.model.load_model(self.xgb_path)

    def xgb_model(self) -> t.Optional[str]:
        """Represent the path of the saved model, if it exists."""
        if os.path.exists(self.xgb_path):
            return self.xgb_path
        return None

    def fit_partial(self, X: pd.DataFrame, y: pd.Series, **kwargs: t.Any) -> None:
        """
        Fit the model partially for online training.

        Args:
            X: The input data to train the model on.
            y: The target data to train the model against.
        """
        self.fit(X, y, **{**kwargs, "xgb_model": self.xgb_model()})

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: t.Any) -> None:
        """
        Fit he model.

        Args:
            X: The input data to train the model on.
            y: The target data to train the model against.
        """
        self.model.fit(X, y, **kwargs)
        self.model.save_model(self.xgb_path)

    def predict(self, X: pd.DataFrame, **kwargs: t.Any) -> np.ndarray:
        """
        Make predictions from the model.

        Args:
            X: The input data to make predictions for.

        Returns:
            The predictions of the model for ``X``.
        """
        return self.model.predict(X, **kwargs)

    def set_trip_data(self, trip_df: pd.DataFrame) -> None:
        """
        Set the internal trip dataframe, used for the ``unconstrained_demand`` API.

        Args:
            trip_df: The dataframe containing the data for a given trip.
        """
        self.trip_df = trip_df

    def unconstrained_demand(self, price: float, day: int) -> float:
        """
        Estimate the unconstrained demand for a given price and a given day.

        Args:
            price: The price to estimate the demand for.
            day: The day to estimate the demand for.

        Raises:
            AttributeError: If ``trip_df`` hasn't been set prior to calling this method.

        Returns:
            The estimated demand for that price and that day. Requires to set the data
            about the trip, by calling ``set_trip_data``. This is required to ensure
            the model knows about the trip characteristics, as well as features about
            the sales prior to ``day``.
        """
        if self.trip_df is None:
            raise AttributeError(
                "Please set `trip_df` by calling `set_trip_data` before calling this "
                "method."
            )
        return self.predict(features.process_trip_at_day(self.trip_df, day, [price]))[0]
