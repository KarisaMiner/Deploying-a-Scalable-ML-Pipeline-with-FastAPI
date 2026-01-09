import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

DATA_PATH = os.path.join("data", "census.csv")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_sample():
    df = pd.read_csv(DATA_PATH)
    return df.head(200)

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_sample():
    df = load_sample()

    """
    # testing process data on sample of census data
    """
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]
    assert encoder is not None
    assert lb is not None


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_sample():
    df = load_sample()
    """
    # train on sample of census data
    """
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0,1})


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_sample():
    df = load_sample()
    """
    # compute_model_metrics returns valid values
    """
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
