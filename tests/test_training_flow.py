"""Runs the real Prefect flow end to end on synthetic prices (no network)."""
import json

import pytest
from prefect.testing.utilities import prefect_test_harness

from flows import training_flow as flow_module
from ml import config, data_ingestion
from ml.drift import run_deepchecks_suite

from conftest import make_prices


@pytest.fixture(scope="module")
def prefect_harness():
    with prefect_test_harness():
        yield


@pytest.fixture
def synthetic_source(monkeypatch):
    prices = make_prices(days=520)

    def fake_fetch(tickers, use_cache=True):
        data_ingestion.last_sources.clear()  # the real fetch_stock_data starts every call this way
        data_ingestion.last_sources.update({t: "yfinance" for t in prices["ticker"].unique()})
        return prices

    monkeypatch.setattr(data_ingestion, "fetch_stock_data", fake_fetch)


@pytest.fixture
def release_metrics(prefect_harness, synthetic_source):
    version = flow_module.training_flow(release=True)
    with open(config.MODELS_DIR / "release" / "metrics.json") as f:
        return version, json.load(f)


def test_the_flow_trains_evaluates_and_saves_a_release(release_metrics):
    version, metrics = release_metrics
    assert version.startswith("version_")
    assert (config.MODELS_DIR / version / "classifier.pkl").exists()
    assert (config.MODELS_DIR / "release" / "classifier.pkl").exists()

    assert metrics["model_version"] == version
    assert metrics["data"]["sources"] == {"AAA": "yfinance", "BBB": "yfinance", "CCC": "yfinance"}
    assert metrics["test_window"]["tickers"] == ["AAA", "BBB", "CCC"]
    assert set(metrics["classification"]["baselines"]) == {"majority", "persistence"}
    assert len(metrics["walk_forward"]["windows"]) >= 1
    assert metrics["validation"]["train_test"]["mode"] in {"DeepChecks", "Manual fallback"}


def test_the_flow_stops_cleanly_when_no_data_arrives(prefect_harness, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(data_ingestion, "fetch_stock_data", lambda tickers, use_cache=True: pd.DataFrame())
    assert flow_module.training_flow() is None
    assert not config.MODELS_DIR.exists()


def test_deepchecks_really_runs_on_the_pinned_stack(split):
    # requirements.txt pins the versions DeepChecks 0.19 needs. If a dependency bump
    # breaks the import again, the pipeline would silently use the manual fallback;
    # this test makes that loud.
    train, test, _ = split
    result = run_deepchecks_suite(train, test)
    assert result["mode"] == "DeepChecks", result
    assert result["checks_passed"] > 0


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: ml.drift.check_feature_drift exists but the flow never calls it, "
    "so no per-feature drift report reaches metrics.json.",
)
def test_the_metrics_file_records_a_feature_drift_report(release_metrics):
    _, metrics = release_metrics
    drift = metrics["validation"]["feature_drift"]
    assert set(drift) == set(config_features())
    assert all("drift_detected" in report for report in drift.values())


def config_features():
    from ml.models import FEATURES

    return FEATURES
