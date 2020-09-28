# wiremind_ds_test
Wiremind Data Science test


## Setup

To extract the dataset into `/data`:

```shell
$ chmod +x extract.sh
$ ./extract.sh
```

## Building the docker image

To build the docker image:

```shell
$ docker build . -t wiremind
```

## Running the docker image

### Preprocessing

To run the preprocessing script:

```shell
$ docker run \
    --rm \
    -v $(pwd)/data:/app/data \
    wiremind \
    scripts/preprocess.py
```

This will read the `dataset.csv` file and split it into a training set and a test set.
The training set will be then processed into the format expected by the model, and the
data will be written into `data/train.csv`. The test set will not be processed further,
and will be written to `data/test.csv`. The train-test split is done over trip ids, to
avoid knowledge of test trips being passed to the trained model.

### Training

To run the training script:

```shell
$ docker run \
    --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/model:/app/model \
    wiremind \
    scripts/train.py
```

This will train a `xgboost` model against the train data. The model will be saved at
`model/xgbmodel`.

### Evaluation

To run the evaluation script:

```shell
$ docker run \
    --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/model:/app/model \
    wiremind \
    scripts/evaluate.py
```

This will compute the MSE and RMSE of the model applied to the test data, and then
display examples of the `unconstrained_demand` method application for each trip of the
test dataset. Before calling the method, please note that it is necessary to call the
`set_trip_data` method with the test dataset for the trip considered; this is necessary
to extract some information, namely:

* The price range of tickets for that trip, to ensure the randomly picked price is
  realistic,
* The trip-specific features, namely the one-hot encoded information about the hour of
  the trip, the day of the week of the trip, and the period during the year of the trip,
* The information about already known bookings for that trip up the the day prior to the
  day considered.
