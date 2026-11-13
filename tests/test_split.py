import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    chronological_split, create_features, risk_thresholds, split_data, to_risk_class,
)

from conftest import make_prices


def test_the_test_window_is_the_last_n_trading_dates(features, split):
    _, test, _ = split
    labelled_dates = np.sort(features.dropna(subset=["future_vol"])["date"].unique())
    assert sorted(test["date"].unique()) == list(labelled_dates[-30:])


def test_every_ticker_is_in_the_test_window(split):
    # Regression test: the original split sorted by ticker and took the last 90 rows,
    # so the "test set" was 90 days of one stock.
    train, test, _ = split
    assert set(test["ticker"]) == set(train["ticker"]) == {"AAA", "BBB", "CCC"}
    assert test.groupby("ticker").size().nunique() == 1


def test_training_ends_before_the_test_window_starts(split):
    train, test, _ = split
    assert train["date"].max() < test["date"].min()


def test_no_training_label_is_computed_from_test_window_returns(features, split):
    # A label looks 5 trading days ahead, so the last training date must sit more than
    # 5 trading dates before the first test date.
    train, test, _ = split
    dates = list(np.sort(features["date"].unique()))
    assert dates.index(test["date"].min()) - dates.index(train["date"].max()) > 5


def test_thresholds_are_learned_from_training_rows_only(features):
    train, test = chronological_split(features, test_days=30, gap_days=5)
    wild = features.copy()
    wild.loc[wild["date"] >= test["date"].min(), "future_vol"] *= 100
    train_again, _ = chronological_split(wild, test_days=30, gap_days=5)
    assert risk_thresholds(train_again) == risk_thresholds(train)


def test_training_classes_are_roughly_thirds(split):
    train, _, _ = split
    shares = train["risk_class"].value_counts(normalize=True)
    assert set(shares.index) == {0, 1, 2}
    assert shares.min() > 0.30


def test_class_boundaries():
    classes = to_risk_class(pd.Series([0.01, 0.02, 0.021, 0.05, 0.051]), (0.02, 0.05))
    assert classes.tolist() == [0, 0, 1, 1, 2]


def test_too_little_history_is_an_error():
    short = create_features(make_prices(days=60))
    with pytest.raises(ValueError):
        split_data(short, test_days=90, gap_days=5)
