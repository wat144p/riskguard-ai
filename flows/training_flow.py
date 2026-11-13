import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to python path to allow imports from 'ml'
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import sklearn
from prefect import flow, get_run_logger, task

from ml import data_ingestion
from ml.config import LABEL_HORIZON_DAYS, TEST_SIZE_DAYS, TICKERS
from ml.drift import check_data_integrity, run_deepchecks_suite
from ml.evaluation import evaluate_models, walk_forward
from ml.feature_engineering import create_features, split_data
from ml.models import save_models, train_models


@task(retries=3, retry_delay_seconds=10)
def get_data_task():
    df = data_ingestion.fetch_stock_data(TICKERS)
    return df, dict(data_ingestion.last_sources)


@task
def validate_data_task(df):
    return check_data_integrity(df)


@task
def feature_engineering_task(df):
    return create_features(df)


@task
def split_data_task(df):
    # Last TEST_SIZE_DAYS trading dates across all tickers, with a gap before them.
    return split_data(df, test_days=TEST_SIZE_DAYS, gap_days=LABEL_HORIZON_DAYS)


@task
def deepchecks_task(train_df, test_df):
    return run_deepchecks_suite(train_df, test_df)


@task
def train_task(train_df):
    return train_models(train_df)


@task
def evaluate_task(models, train_df, test_df, thresholds):
    return evaluate_models(models, train_df, test_df, thresholds)


@task
def walk_forward_task(df_features):
    return walk_forward(df_features)


@task
def save_task(models, metrics, release):
    return save_models(models, metrics, release=release)


@task(name="notify_completion")
def notify_completion(version: str):
    """Placeholder for a Slack or email notification; it only prints."""
    print(f"\n[NOTIFICATION] Pipeline completed. Model version: {version}\n")


@flow(name="Stock Risk Training Flow")
def training_flow(release: bool = False):
    """
    Ingestion -> integrity check -> features -> chronological split -> DeepChecks
    -> training -> evaluation against baselines -> walk-forward evaluation -> save.

    release=True also copies the run to models/release/, the folder the repo ships.
    """
    logger = get_run_logger()

    raw_df, sources = get_data_task()
    if raw_df.empty:
        logger.error("No data could be fetched from any source. Stopping.")
        return None

    integrity = validate_data_task(raw_df)
    if not integrity["passed"]:
        logger.warning(f"Data integrity warning: {integrity}")

    df_features = feature_engineering_task(raw_df)
    train_df, test_df, thresholds = split_data_task(df_features)

    deepchecks_result = deepchecks_task(train_df, test_df)
    if deepchecks_result["passed"]:
        logger.info(f"Train/test validation passed ({deepchecks_result['mode']}).")
    else:
        logger.warning(f"Train/test validation flagged: {deepchecks_result}")

    models = train_task(train_df)

    metrics = evaluate_task(models, train_df, test_df, thresholds)
    metrics["walk_forward"] = walk_forward_task(df_features)
    metrics["validation"] = {"integrity_passed": integrity["passed"], "train_test": deepchecks_result}
    metrics["data"] = {
        "sources": sources,
        "rows": int(len(raw_df)),
        "first_date": str(pd.Timestamp(raw_df["date"].min()).date()),
        "last_date": str(pd.Timestamp(raw_df["date"].max()).date()),
    }
    metrics["run"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }

    version = save_task(models, metrics, release)
    notify_completion(version)

    clf = metrics["classification"]
    logger.info(
        f"Done: {version}. Test accuracy {clf['accuracy']:.3f} "
        f"(majority {clf['baselines']['majority']:.3f}, persistence {clf['baselines']['persistence']:.3f})."
    )
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the RiskGuard models.")
    parser.add_argument("--release", action="store_true", help="also copy this run to models/release/")
    training_flow(release=parser.parse_args().release)
