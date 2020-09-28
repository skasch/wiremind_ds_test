#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The evaluate script.

Created by Romain Mondon-Cancel on 2020-09-27 16:15:36
"""

import logging
import math
import os

import numpy as np
import pandas as pd
import sklearn.metrics

import wiremind as wm

logging.getLogger().setLevel(logging.INFO)

DATA_PATH = os.path.realpath("data")
MODEL_PATH = os.path.realpath("model")
TEST_PATH = os.path.join(DATA_PATH, "test.csv")
XGB_MODEL_PATH = os.path.join(MODEL_PATH, "xgb_model.pkl")

test_df = pd.read_csv(TEST_PATH)
processed_test_df = wm.features.process_df(test_df)
processed_test_X, processed_test_y = wm.features.split_X_y(processed_test_df)

model = wm.model.DemandModel(xgb_path=XGB_MODEL_PATH)
predicted_demand = model.predict(processed_test_X)
mse = sklearn.metrics.mean_squared_error(processed_test_y, predicted_demand)
print("Metrics on a random sample of trips with random prices:")
print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {math.sqrt(mse)}")

print("Example calling the model on individual trips.")
for trip_id, trip_df in test_df.groupby("trip_id"):
    model.set_trip_data(trip_df)
    price = wm.features.generate_prices(trip_df.price.min(), trip_df.price.max(), 1)[0]
    day = np.random.choice(trip_df.sale_day.unique())
    expected_demand = model.unconstrained_demand(price, day)
    actual_demand = trip_df.loc[
        lambda df: (df.sale_day == day) & (df.price >= price)
    ].shape[0]
    print(
        f"The expected demand for trip {trip_id} on day {day} with price {price} is "
        f"{expected_demand}. The actual demand was {actual_demand}."
    )
