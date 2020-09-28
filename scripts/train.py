#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The train script.

Created by Romain Mondon-Cancel on 2020-09-27 11:01:09
"""

import logging
import os

import pandas as pd
import sklearn.model_selection

import wiremind as wm

logging.getLogger().setLevel(logging.INFO)

DATA_PATH = os.path.realpath("data")
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
MODEL_PATH = os.path.realpath("model")
XGB_MODEL_PATH = os.path.join(MODEL_PATH, "xgb_model.pkl")
RESET_MODEL = True
EVAL_RATIO = 0.05

(train_X, train_y), (xval_X, xval_y) = map(
    wm.features.split_X_y,
    sklearn.model_selection.train_test_split(
        pd.read_csv(TRAIN_PATH).sample(frac=1), test_size=EVAL_RATIO
    ),
)
print(f"Training data loaded successfully from {TRAIN_PATH}.")

if RESET_MODEL and os.path.exists(XGB_MODEL_PATH):
    os.remove(XGB_MODEL_PATH)
    print(f"Removed existing model at {XGB_MODEL_PATH}.")

model = wm.model.DemandModel(xgb_path=XGB_MODEL_PATH)
print("Created demand model.")

model.fit(train_X, train_y, eval_set=[(xval_X, xval_y)], verbose=True)
print(f"Model trained successfully, and saved to {XGB_MODEL_PATH}.")

print("Feature importance:")
print(dict(zip(train_X.columns, model.model.feature_importances_)))
