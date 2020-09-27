#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The preprocess script.

Created by Romain Mondon-Cancel on 2020-09-27 10:25:55
"""

import logging
import os

import wiremind as wm  # noqa: E402

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_PATH = os.path.realpath("data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset.csv")
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH = os.path.join(DATA_PATH, "test.csv")

df = wm.data.read_csv(DATASET_PATH).pipe(wm.features.add_features)
logger.info(f"Data loaded successfully with initial features from {DATASET_PATH}.")
train_df, test_df = wm.features.train_test_split(df)
logger.info("Data split into train and test datasets.")
processed_train_df = wm.features.process_df(train_df)
logger.info("Train data successfully processed.")
processed_test_df = wm.features.process_df(test_df)
logger.info("Test data successfully processed.")
processed_train_df.to_csv(TRAIN_PATH, index=False)
logger.info(f"Train data successfully saved into {TRAIN_PATH}.")
processed_test_df.to_csv(TEST_PATH, index=False)
logger.info(f"Test data successfully saved into {TEST_PATH}.")
