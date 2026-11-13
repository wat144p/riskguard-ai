import os
import time

import pandas as pd
import pytest

from ml import data_ingestion
from ml.data_ingestion import SourceUnavailable, fetch_stock_data

REAL_FROM_YFINANCE = data_ingestion._from_yfinance  # captured before the autouse fixture patches it
COLUMNS = ["date", "open", "high", "low", "close", "volume", "ticker"]


def yahoo_frame(ticker="AAPL"):
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": 1.0, "high": 2.0, "low": 0.5, "close": [10.0, 11.0, 12.0], "volume": 100, "ticker": ticker}
    )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def yahoo(monkeypatch):
    calls = []

    def fake(ticker):
        calls.append(ticker)
        return yahoo_frame(ticker)

    monkeypatch.setattr(data_ingestion, "_from_yfinance", fake)
    return calls


def test_without_an_api_key_only_yfinance_is_used(yahoo):
    # requests.get still raises (autouse fixture), so this also proves Alpha Vantage is not called.
    df = fetch_stock_data(["AAPL", "MSFT"], use_cache=False)
    assert yahoo == ["AAPL", "MSFT"]
    assert list(df.columns) == COLUMNS
    assert data_ingestion.last_sources == {"AAPL": "yfinance", "MSFT": "yfinance"}


@pytest.mark.parametrize(
    "payload",
    [
        {"Information": "This is a premium endpoint."},
        {"Note": "API call frequency exceeded."},
        {"Error Message": "Invalid API call."},
        {},
    ],
)
def test_alpha_vantage_refusals_fall_back_to_yfinance(monkeypatch, yahoo, payload):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(data_ingestion.requests, "get", lambda *a, **k: FakeResponse(payload))
    df = fetch_stock_data(["AAPL"], use_cache=False)
    assert yahoo == ["AAPL"]
    assert len(df) == 3
    assert data_ingestion.last_sources == {"AAPL": "yfinance"}


def test_a_network_error_falls_back_to_yfinance(monkeypatch, yahoo):
    def boom(*args, **kwargs):
        raise data_ingestion.requests.ConnectionError("down")

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(data_ingestion.requests, "get", boom)
    assert len(fetch_stock_data(["AAPL"], use_cache=False)) == 3


def test_alpha_vantage_prices_are_split_adjusted(monkeypatch):
    # A 2-for-1 split between the two days: the raw close halves, the adjusted close does not.
    today = pd.Timestamp.now(tz="UTC").normalize()
    day1 = (today - pd.Timedelta(days=2)).date().isoformat()
    day2 = (today - pd.Timedelta(days=1)).date().isoformat()

    def row(open_, close, adjusted):
        return {
            "1. open": str(open_), "2. high": str(open_), "3. low": str(open_), "4. close": str(close),
            "5. adjusted close": str(adjusted), "6. volume": "10",
            "7. dividend amount": "0", "8. split coefficient": "1",
        }

    payload = {"Time Series (Daily)": {day1: row(200, 200, 100), day2: row(100, 101, 101)}}
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(data_ingestion.requests, "get", lambda *a, **k: FakeResponse(payload))

    df = fetch_stock_data(["AAPL"], use_cache=False)
    assert df["close"].tolist() == [100.0, 101.0]  # no fake -50% return
    assert df["open"].tolist() == [100.0, 100.0]
    assert data_ingestion.last_sources == {"AAPL": "alpha_vantage"}


def test_a_fresh_cache_is_used_and_a_stale_one_is_refetched(yahoo):
    fetch_stock_data(["AAPL"])  # fills the cache
    fetch_stock_data(["AAPL"])  # served from it
    assert yahoo == ["AAPL"]
    assert data_ingestion.last_sources == {"AAPL": "cache"}

    cache_file = data_ingestion.DATA_DIR / "AAPL.csv"
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(cache_file, (two_days_ago, two_days_ago))
    fetch_stock_data(["AAPL"])
    assert yahoo == ["AAPL", "AAPL"]


def test_a_ticker_with_no_data_is_skipped_not_fatal(monkeypatch):
    def fake(ticker):
        if ticker == "NOPE":
            raise SourceUnavailable("yfinance returned no rows")
        return yahoo_frame(ticker)

    monkeypatch.setattr(data_ingestion, "_from_yfinance", fake)
    df = fetch_stock_data(["NOPE", "AAPL"], use_cache=False)
    assert set(df["ticker"]) == {"AAPL"}
    assert data_ingestion.last_sources["NOPE"] == "failed"


def test_nothing_fetched_returns_an_empty_frame(monkeypatch):
    def fail(ticker):
        raise SourceUnavailable("down")

    monkeypatch.setattr(data_ingestion, "_from_yfinance", fail)
    assert fetch_stock_data(["AAPL"], use_cache=False).empty


def test_the_api_key_is_never_printed(monkeypatch, yahoo, capsys):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(data_ingestion.requests, "get", lambda *a, **k: FakeResponse({"Note": "slow down"}))
    fetch_stock_data(["AAPL"], use_cache=False)
    assert "SECRET-KEY-123" not in capsys.readouterr().out


def test_yfinance_rows_are_standardised(monkeypatch):
    import yfinance

    history = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10], "Dividends": [0.0]},
        index=pd.DatetimeIndex(["2024-01-02"], name="Date", tz="America/New_York"),
    )

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period):
            return history

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    df = REAL_FROM_YFINANCE("AAPL")
    assert list(df.columns) == COLUMNS
    assert str(df["date"].dt.tz) == "UTC"


def test_yfinance_returning_nothing_is_reported(monkeypatch):
    import yfinance

    class EmptyTicker:
        def __init__(self, symbol):
            pass

        def history(self, period):
            return pd.DataFrame()

    monkeypatch.setattr(yfinance, "Ticker", EmptyTicker)
    with pytest.raises(SourceUnavailable):
        REAL_FROM_YFINANCE("AAPL")


def test_a_row_without_a_closing_price_is_dropped(monkeypatch):
    # Seen live on 13 Nov 2026: yfinance returned the latest NVDA day with a volume
    # but no prices, and /predict_risk failed with "Input X contains NaN".
    import yfinance

    history = pd.DataFrame(
        {"Open": [1.0, None], "High": [2.0, None], "Low": [0.5, None], "Close": [1.5, None], "Volume": [10, 99]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date", tz="America/New_York"),
    )

    class PartialTicker:
        def __init__(self, symbol):
            pass

        def history(self, period):
            return history

    monkeypatch.setattr(yfinance, "Ticker", PartialTicker)
    df = REAL_FROM_YFINANCE("NVDA")
    assert len(df) == 1
    assert df["close"].notna().all()
