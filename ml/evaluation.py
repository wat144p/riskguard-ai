"""Evaluation: the model's scores next to the baselines it has to beat.

A 3-class accuracy number means nothing on its own. Two cheap baselines are reported
with every result:
  - majority:    always guess the most common class in the test window
  - persistence: assume next week's volatility equals last week's (bucket the current
                 5-day volatility with the same thresholds). Volatility clusters, so
                 this one-line rule is hard to beat.
A logistic regression on the same features is trained as a reference model, to show
whether gradient boosting is actually earning its complexity.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .feature_engineering import add_risk_class, risk_thresholds, to_risk_class
from .models import train_models

CLASS_LABELS = [0, 1, 2]


def baseline_accuracies(test_df: pd.DataFrame, thresholds: tuple[float, float]) -> dict:
    y_true = test_df["risk_class"]
    return {
        "majority": float(y_true.value_counts(normalize=True).max()),
        "persistence": float(accuracy_score(y_true, to_risk_class(test_df["volatility_5d"], thresholds))),
    }


def reference_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> float:
    """Accuracy of a plain logistic regression trained on the same rows and features."""
    reference = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    reference.fit(train_df[features], train_df["risk_class"])
    return float(accuracy_score(test_df["risk_class"], reference.predict(test_df[features])))


def evaluate_models(
    models: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, thresholds: tuple[float, float]
) -> dict:
    """Scores the trained models on the held-out window."""
    features = models["features"]
    X_test = test_df[features]
    y_clf = test_df["risk_class"]
    y_reg = test_df["target_return_next_day"]

    clf_pred = models["classifier"].predict(X_test)
    reg_pred = models["regressor"].predict(X_test)

    per_ticker = {
        ticker: float(accuracy_score(group["risk_class"], models["classifier"].predict(group[features])))
        for ticker, group in test_df.groupby("ticker")
    }

    return {
        "test_window": {
            "rows": int(len(test_df)),
            "tickers": sorted(test_df["ticker"].unique().tolist()),
            "first_date": str(pd.Timestamp(test_df["date"].min()).date()),
            "last_date": str(pd.Timestamp(test_df["date"].max()).date()),
        },
        "train_window": {
            "rows": int(len(train_df)),
            "first_date": str(pd.Timestamp(train_df["date"].min()).date()),
            "last_date": str(pd.Timestamp(train_df["date"].max()).date()),
        },
        "risk_thresholds": {"low_below": thresholds[0], "high_above": thresholds[1]},
        "classification": {
            "accuracy": float(accuracy_score(y_clf, clf_pred)),
            "f1_weighted": float(f1_score(y_clf, clf_pred, average="weighted")),
            "train_accuracy": float(
                accuracy_score(train_df["risk_class"], models["classifier"].predict(train_df[features]))
            ),
            "baselines": baseline_accuracies(test_df, thresholds),
            "logistic_regression_reference": reference_accuracy(train_df, test_df, features),
            "per_ticker_accuracy": per_ticker,
            "confusion_matrix": {
                "labels": config.RISK_LEVELS,
                "rows_true_cols_predicted": confusion_matrix(y_clf, clf_pred, labels=CLASS_LABELS).tolist(),
            },
        },
        "regression": {
            "r2": float(r2_score(y_reg, reg_pred)),
            "train_r2": float(
                r2_score(train_df["target_return_next_day"], models["regressor"].predict(train_df[features]))
            ),
            "rmse": float(np.sqrt(mean_squared_error(y_reg, reg_pred))),
            "mae": float(mean_absolute_error(y_reg, reg_pred)),
        },
    }


def walk_forward(
    df_features: pd.DataFrame,
    n_windows: int = config.WALK_FORWARD_WINDOWS,
    test_days: int = config.TEST_SIZE_DAYS,
    gap_days: int = config.LABEL_HORIZON_DAYS,
) -> dict:
    """Repeats train-then-test over consecutive windows, oldest first.

    One 90-day window is a small sample and can flatter or punish a model by luck.
    Each window here trains only on data that ends `gap_days` before the window starts,
    with its own thresholds, exactly like the final split.
    """
    labelled = df_features.dropna(subset=["future_vol", "target_return_next_day"])
    dates = np.sort(labelled["date"].unique())
    windows = []
    for k in range(n_windows, 0, -1):
        start = len(dates) - k * test_days
        end = start + test_days  # exclusive
        if start - gap_days - 1 < 250:  # need about a year of history to train on
            continue
        train = labelled[labelled["date"] <= dates[start - gap_days - 1]]
        test = labelled[(labelled["date"] >= dates[start]) & (labelled["date"] <= dates[end - 1])]
        thresholds = risk_thresholds(train)
        train, test = add_risk_class(train, thresholds), add_risk_class(test, thresholds)
        models = train_models(train)
        features = models["features"]
        windows.append(
            {
                "first_date": str(pd.Timestamp(dates[start]).date()),
                "last_date": str(pd.Timestamp(dates[end - 1]).date()),
                "rows": int(len(test)),
                "accuracy": float(accuracy_score(test["risk_class"], models["classifier"].predict(test[features]))),
                **baseline_accuracies(test, thresholds),
                "logistic_regression_reference": reference_accuracy(train, test, features),
                "regression_r2": float(
                    r2_score(test["target_return_next_day"], models["regressor"].predict(test[features]))
                ),
            }
        )

    def mean(key):
        return float(np.mean([w[key] for w in windows])) if windows else None

    return {
        "windows": windows,
        "mean": {
            key: mean(key)
            for key in ("accuracy", "majority", "persistence", "logistic_regression_reference", "regression_r2")
        },
    }
