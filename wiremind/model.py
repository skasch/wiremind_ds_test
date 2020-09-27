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
        self.train_info: t.Optional[pd.Series] = None
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

    def fit_train(self, train_df: pd.DataFrame) -> None:
        self.model.fit(
            processed_train_df[X_cols],
            processed_train_df[["demand"]],
            xgb_model=self.xgb_model(),
        )
        self.model.save_model(self.xgb_path)

    def fit(self, df_groups: t.Iterable[t.Tuple[str, pd.DataFrame]]) -> None:
        for train_id, train_df in df_groups:
            logger.info(f"Fitting model on data for train id {train_id}.")
            self.fit_train(train_df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pass
