import pandas as pd
import numpy as np

def check_data_integrity(df: pd.DataFrame) -> dict:
    """
    Checks for basic data integrity issues.
    """
    report = {
        "passed": True,
        "missing_values": {},
        "empty_ticker_data": []
    }
    
    if df.empty:
        report["passed"] = False
        report["error"] = "DataFrame is empty"
        return report
        
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        report["missing_values"] = nan_counts[nan_counts > 0].to_dict()
        # Not a hard failure unless key columns are missing, but let's log it
    
    # Check for tickers with too little data
    ticker_counts = df["ticker"].value_counts()
    for ticker, count in ticker_counts.items():
        if count < 50:
            report["empty_ticker_data"].append(ticker)
            report["passed"] = False
            
    return report

def check_feature_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, features: list) -> dict:
    """
    detects drift by comparing mean of features.
    Simple method: if mean differs by > 2 standard deviations of training data.
    """
    drift_report = {}
    
    for feature in features:
        train_mean = train_df[feature].mean()
        train_std = train_df[feature].std()
        
        new_mean = new_df[feature].mean()
        
        # Avoid division by zero
        if train_std == 0:
            z_score = 0 if new_mean == train_mean else 999
        else:
            z_score = abs(new_mean - train_mean) / train_std
            
        drift_report[feature] = {
            "train_mean": train_mean,
            "new_mean": new_mean,
            "z_score": z_score,
            "drift_detected": z_score > 3 # 3 sigma
        }
        
    return drift_report

def run_deepchecks_suite(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Runs the DeepChecks train-test validation suite on the model's inputs and label.

    DeepChecks 0.19 only imports on the pinned stack in requirements.txt (numpy 1.x,
    scikit-learn < 1.6, setuptools < 81). If it cannot be imported or crashes, a small
    manual check runs instead so the pipeline still finishes; the returned "mode" says
    which of the two actually ran, and it is written to metrics.json.
    """
    from .models import FEATURES

    columns = FEATURES + ["ticker", "risk_class"]
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import train_test_validation

        ds_train = Dataset(train_df[columns], label="risk_class", cat_features=["ticker"])
        ds_test = Dataset(test_df[columns], label="risk_class", cat_features=["ticker"])
        result = train_test_validation().run(train_dataset=ds_train, test_dataset=ds_test)

        failures = [check.get_header() for check in result.get_not_passed_checks()]
        return {
            "mode": "DeepChecks",
            "passed": len(failures) == 0,
            "failures": failures,
            "checks_passed": len(result.get_passed_checks()),
        }

    except Exception as e:
        print(f"DeepChecks did not run ({type(e).__name__}); using the manual fallback check.")
        failures = []
        if train_df[columns].isna().sum().sum() > 0:
            failures.append("Null values in training data")
        if len(train_df) < 100:
            failures.append("Fewer than 100 training rows")
        return {
            "mode": "Manual fallback",
            "passed": len(failures) == 0,
            "failures": failures,
            "reason": type(e).__name__,
        }
