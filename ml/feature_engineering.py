import pandas as pd
import numpy as np
from .config import LABEL_HORIZON_DAYS, TEST_SIZE_DAYS

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for time-series analysis.
    Assumes df has columns: 'ticker', 'date', 'close' etc.
    """
    df = df.copy()
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping feature creation.")
        return df

    df = df.sort_values(["ticker", "date"])

    # Calculate daily returns
    df["return"] = df.groupby("ticker")["close"].pct_change()
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df.groupby("ticker")["return"].shift(lag)
    
    # Rolling statistics
    # Volatility (Std Dev of returns)
    df["volatility_5d"] = df.groupby("ticker")["return"].rolling(window=5).std().reset_index(0, drop=True)
    df["volatility_20d"] = df.groupby("ticker")["return"].rolling(window=20).std().reset_index(0, drop=True)
    
    # Rolling mean
    df["ma_5d"] = df.groupby("ticker")["close"].rolling(window=5).mean().reset_index(0, drop=True)
    df["ma_20d"] = df.groupby("ticker")["close"].rolling(window=20).mean().reset_index(0, drop=True)
    
    # Relative to MA
    df["price_vs_ma20"] = (df["close"] - df["ma_20d"]) / df["ma_20d"]

    # RSI (Relative Strength Index)
    delta = df.groupby("ticker")["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.groupby(df["ticker"]).rolling(window=14).mean().reset_index(0, drop=True)
    avg_loss = loss.groupby(df["ticker"]).rolling(window=14).mean().reset_index(0, drop=True)
    
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid div by zero
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50) # Neutral fill

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df.groupby("ticker")["close"].ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
    ema_26 = df.groupby("ticker")["close"].ewm(span=26, adjust=False).mean().reset_index(0, drop=True)
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("ticker")["macd"].ewm(span=9, adjust=False).mean().reset_index(0, drop=True)

    # Targets
    # Regression target: next day's return.
    df["target_return_next_day"] = df.groupby("ticker")["return"].shift(-1)

    # Classification target: volatility of the NEXT 5 daily returns (t+1 .. t+5).
    # Every feature above only looks backwards, so features and label never share a day.
    df["future_vol"] = (
        df.groupby("ticker")["return"]
        .transform(lambda r: r.rolling(window=LABEL_HORIZON_DAYS).std().shift(-LABEL_HORIZON_DAYS))
    )

    # Drop the warm-up rows where lags and rolling windows are not filled yet.
    # The last 5 rows of each ticker keep a NaN future_vol: the API predicts on them,
    # and the split below drops them from training and testing.
    df = df.dropna(subset=["return_lag5", "volatility_20d"])
    return df


def chronological_split(
    df: pd.DataFrame, test_days: int = TEST_SIZE_DAYS, gap_days: int = LABEL_HORIZON_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by calendar date, across all tickers at once.

    Test  = every row whose date is one of the last `test_days` trading dates.
    Train = every row dated more than `gap_days` trading dates before the first test date.

    The gap matters because a label looks `gap_days` ahead: without it, the last
    training labels would be computed from returns inside the test window.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    labelled = df.dropna(subset=["future_vol", "target_return_next_day"])
    dates = np.sort(labelled["date"].unique())
    if len(dates) <= test_days + gap_days:
        raise ValueError(f"Need more than {test_days + gap_days} trading dates, got {len(dates)}.")

    first_test_date = dates[-test_days]
    last_train_date = dates[-test_days - gap_days - 1]
    train_df = labelled[labelled["date"] <= last_train_date].copy()
    test_df = labelled[labelled["date"] >= first_test_date].copy()
    return train_df, test_df


def risk_thresholds(train_df: pd.DataFrame) -> tuple[float, float]:
    """The two volatility cut-offs between Low / Medium / High, from training rows only."""
    return (
        float(train_df["future_vol"].quantile(0.33)),
        float(train_df["future_vol"].quantile(0.66)),
    )


def to_risk_class(volatility: pd.Series, thresholds: tuple[float, float]) -> np.ndarray:
    """0 = Low, 1 = Medium, 2 = High."""
    low, high = thresholds
    return np.where(volatility <= low, 0, np.where(volatility <= high, 1, 2))


def add_risk_class(df: pd.DataFrame, thresholds: tuple[float, float]) -> pd.DataFrame:
    df = df.copy()
    df["risk_class"] = to_risk_class(df["future_vol"], thresholds)
    return df


def split_data(
    df: pd.DataFrame, test_days: int = TEST_SIZE_DAYS, gap_days: int = LABEL_HORIZON_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Chronological split, then labels both halves with thresholds learned on train."""
    train_df, test_df = chronological_split(df, test_days, gap_days)
    thresholds = risk_thresholds(train_df)
    return add_risk_class(train_df, thresholds), add_risk_class(test_df, thresholds), thresholds
