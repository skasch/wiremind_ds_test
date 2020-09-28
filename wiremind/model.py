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
    def __init__(
        self,
        xgb_path: str = os.path.join("model", "xgb_model.pkl"),
        xgb_args: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        if xgb_args is None:
            xgb_args = {}
        self.xgb_path = xgb_path
        self.trip_df: t.Optional[pd.DataFrame] = None
        self.initialize_model(**xgb_args)

    def initialize_model(self, **xgb_args: t.Any) -> None:
        logger.info("Initializing model.")
        self.model = xgb.XGBRegressor(**xgb_args)
        if os.path.exists(self.xgb_path):
            logger.info(f"Found existing model at {self.xgb_path}; loading model.")
            self.model.load_model(self.xgb_path)

    def xgb_model(self) -> t.Optional[str]:
        if os.path.exists(self.xgb_path):
            return self.xgb_path
        return None

    def fit_partial(self, X: pd.DataFrame, y: pd.Series, **kwargs: t.Any) -> None:
        self.fit(X, y, **{**kwargs, "xgb_model": self.xgb_model()})

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: t.Any) -> None:
        self.model.fit(X, y, **kwargs)
        self.model.save_model(self.xgb_path)

    def predict(self, X: pd.DataFrame, **kwargs: t.Any) -> np.ndarray:
        return self.model.predict(X, **kwargs)

    def set_trip_data(self, trip_df: pd.DataFrame) -> None:
        self.trip_df = trip_df

    def unconstrained_demand(self, price: float, day: int) -> float:
        if self.trip_df is None:
            raise AttributeError(
                "Please set `trip_df` by calling `set_trip_data` before calling this "
                "method."
            )
        return self.predict(features.process_trip_at_day(self.trip_df, day, [price]))[0]
