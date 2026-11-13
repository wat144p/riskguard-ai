"""Shared fixtures. No test touches the network or the real models/ and data/ folders."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import config, data_ingestion  # noqa: E402
from ml.feature_engineering import create_features, split_data  # noqa: E402
from ml.models import train_models  # noqa: E402


def make_prices(tickers=("AAA", "BBB", "CCC"), days=420, seed=0) -> pd.DataFrame:
    """Random-walk prices whose volatility switches between calm and wild every 30 days,
    so that "next week's risk" is something a model can actually learn."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days, tz="UTC")
    frames = []
    for i, ticker in enumerate(tickers):
        daily_vol = np.where((np.arange(days) // 30 + i) % 2 == 0, 0.005, 0.03)
        close = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
                    "close": close, "volume": 1_000_000, "ticker": ticker,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Temp folders instead of models/ and data/, no API key, no sleeping, no network."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(data_ingestion, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(data_ingestion.time, "sleep", lambda seconds: None)

    def no_network(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(data_ingestion.requests, "get", no_network)
    monkeypatch.setattr(data_ingestion, "_from_yfinance", no_network)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def features(prices):
    return create_features(prices)


@pytest.fixture
def split(features):
    """(train, test, thresholds) with a 30-day test window."""
    return split_data(features, test_days=30, gap_days=5)


@pytest.fixture
def trained(split):
    train, _, _ = split
    return train_models(train)
