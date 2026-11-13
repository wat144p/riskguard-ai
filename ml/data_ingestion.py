"""Daily price history for a list of tickers.

Each ticker is looked up in this order:
  1. the local CSV cache in data/ (if younger than 24 hours)
  2. Alpha Vantage (only when ALPHAVANTAGE_API_KEY is set)
  3. yfinance (when there is no key, or Alpha Vantage fails for any reason)

Every source returns the same columns: date, open, high, low, close, volume, ticker.
Prices are split- and dividend-adjusted in both sources, so a stock split does not
show up as a fake -90% daily return.
"""
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import DATA_DIR, HISTORY_YEARS

load_dotenv()

CACHE_MAX_AGE_S = 24 * 60 * 60
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_PAUSE_S = 12  # the free tier allows about 5 calls a minute
REQUEST_TIMEOUT_S = 30
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Which source served each ticker on the most recent fetch_stock_data() call.
# The training flow writes this into the metrics file.
last_sources: Dict[str, str] = {}


class SourceUnavailable(Exception):
    """A data source could not return usable prices for a ticker."""


def _standardize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for col in PRICE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = ticker
    # yfinance sometimes returns a half-filled row for the current day (volume but no
    # prices). A row without a close would turn into a fake 0% return and a NaN feature.
    df = df.dropna(subset=["close"])
    return df.sort_values("date").reset_index(drop=True)


def _from_cache(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > CACHE_MAX_AGE_S:
        print(f"Cache for {ticker} is older than 24h, refetching.")
        return None
    try:
        return _standardize(pd.read_csv(path), ticker)
    except Exception:
        print(f"Cache for {ticker} is unreadable, refetching.")
        return None


def _from_alpha_vantage(ticker: str, api_key: str) -> pd.DataFrame:
    try:
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": api_key,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SourceUnavailable(f"request failed ({type(exc).__name__})") from exc

    series = payload.get("Time Series (Daily)")
    if not series:
        # Rate-limit notes, premium-endpoint notices and bad symbols all arrive as a
        # JSON object with a single message key instead of the time series.
        reason = next((k for k in ("Information", "Note", "Error Message") if k in payload), "no time series")
        raise SourceUnavailable(reason)

    raw = pd.DataFrame.from_dict(series, orient="index").apply(pd.to_numeric, errors="coerce")
    # Put open/high/low/close on the adjusted basis (the same basis yfinance uses).
    factor = raw["5. adjusted close"] / raw["4. close"]
    df = pd.DataFrame(
        {
            "date": raw.index,
            "open": raw["1. open"] * factor,
            "high": raw["2. high"] * factor,
            "low": raw["3. low"] * factor,
            "close": raw["5. adjusted close"],
            "volume": raw["6. volume"],
        }
    )
    df = _standardize(df, ticker)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=HISTORY_YEARS)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def _from_yfinance(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period=f"{HISTORY_YEARS}y")
    except Exception as exc:
        raise SourceUnavailable(f"yfinance error ({type(exc).__name__})") from exc
    if hist is None or hist.empty:
        raise SourceUnavailable("yfinance returned no rows")
    hist = hist.reset_index().rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    return _standardize(hist, ticker)


def fetch_ticker(ticker: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """Returns one ticker's history, or None if every source failed."""
    if use_cache:
        cached = _from_cache(ticker)
        if cached is not None:
            last_sources[ticker] = "cache"
            return cached

    df = None
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key:
        try:
            df = _from_alpha_vantage(ticker, api_key)
            last_sources[ticker] = "alpha_vantage"
        except SourceUnavailable as exc:
            print(f"Alpha Vantage unavailable for {ticker} ({exc}); falling back to yfinance.")
        finally:
            time.sleep(ALPHA_VANTAGE_PAUSE_S)

    if df is None:
        try:
            df = _from_yfinance(ticker)
            last_sources[ticker] = "yfinance"
        except SourceUnavailable as exc:
            print(f"No data for {ticker}: {exc}")
            last_sources[ticker] = "failed"
            return None

    if use_cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / f"{ticker}.csv", index=False)
    return df


def fetch_stock_data(tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
    """Returns one combined DataFrame for all tickers that could be fetched.

    Columns: date, open, high, low, close, volume, ticker. Empty if nothing was fetched.
    """
    last_sources.clear()
    frames = [df for df in (fetch_ticker(t, use_cache) for t in tickers) if df is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
