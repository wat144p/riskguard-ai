import numpy as np
import pandas as pd

from ml.feature_engineering import create_features
from ml.models import FEATURES

from conftest import make_prices


def test_future_vol_is_the_std_of_the_next_five_returns(features):
    one = features[features.ticker == "AAA"].reset_index(drop=True)
    i = 100
    expected = one["return"].iloc[i + 1 : i + 6].std()
    assert np.isclose(one["future_vol"].iloc[i], expected)


def test_the_label_never_borrows_returns_from_the_next_ticker(features):
    # The last 5 rows of every ticker have no 5 future returns, so no label.
    for _, group in features.groupby("ticker"):
        assert group["future_vol"].tail(5).isna().all()
        assert group["future_vol"].iloc[:-5].notna().all()


def test_features_only_look_backwards():
    prices = make_prices()
    changed = prices.copy()
    cutoff = prices["date"].unique()[300]
    later = changed["date"] > cutoff
    changed.loc[later, "close"] = changed.loc[later, "close"] * 3  # rewrite the future

    before = create_features(prices)
    after = create_features(changed)
    pd.testing.assert_frame_equal(
        before.loc[before["date"] <= cutoff, FEATURES],
        after.loc[after["date"] <= cutoff, FEATURES],
    )


def test_warm_up_rows_are_dropped(features):
    assert features[FEATURES].notna().all().all()


def test_empty_input_returns_empty():
    assert create_features(pd.DataFrame()).empty
