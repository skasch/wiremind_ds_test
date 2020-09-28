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

## Improvements

Here are a couple points I thought about or tried to improve the model:

### Feature engineering

* The period of the year used is a rough segmentation to take yearly seasonality into
  account; there is a lot of potential improvements to work on there, for example to
  take holidays into account.
* Additional, external information about events happening in one of the destination
  stations would probably also be very useful to improve the quality of the model.
* I did not take cancellations into account in this first design; A first idea that came
  to mind but that I did not implement at this point was to add columns to describe the
  total number of tickets cancelled up to a given day, as well as the total number of
  tickets cancelled above the target price up to a given day.
* About ticket specific features, I used a naive, uniform distribution over the price
  range to generate random prices, but that's probably not ideal; there are probably
  more interesting distributions to use, and notably distributions that depend on the
  day considered.
* Talking about distribution, representing the past demand by only considering the total
  number of tickets sold and the total number of tickets sold above the price is a bit
  poor; there are certainly more interesting approaches.
* I made a strong assumption, but this seemed to be in line with the description of the
  problem. I assumed we know beforehand the price range of tickets before inferring the
  demand. I extract this information from the price at which the tickets were sold,
  which means this approach is invalid if this assumption is false.
* May hours are useless at this point, as there is no train circulating at night. This
  means we have useless columns, but the upside is that the model can easily be used
  as is if trains are scheduled in the future without having to retrain from scratch.

### Model fine-tuning

* I did not spend much time fine-tuning the model. I used a simple XGBoost regressor,
  which seemed to be a good candidate to yield good performance without too much effort
  on that front.
* I did try to use the RMSLE (Root Mean Squared Log Error), as I believe a
  multiplicative error makes more sense than an additive error, but the model could
  still predict demands below `-1`, which caused the loss to become `NaN`. That's why
  I'm using a simple RMSE as the objective, even though I don't think it represents the
  best target for the problem.
* There are a lot of other hyper-parameters that could be optimized even if we decide
  to keep using the same model, and a whole lot more if we decide to explore other
  kinds of models.
