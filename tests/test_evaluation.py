import pandas as pd
import pytest

from ml.evaluation import baseline_accuracies, evaluate_models, walk_forward
from ml.models import load_latest_metrics, load_latest_models, save_models


def test_baselines_on_a_hand_made_example():
    thresholds = (0.01, 0.02)
    test = pd.DataFrame(
        {
            # true classes: 3 High, 1 Low -> always guessing High scores 0.75
            "risk_class": [2, 2, 2, 0],
            # current volatility buckets to High, High, Medium, Low -> 3 of 4 right
            "volatility_5d": [0.03, 0.05, 0.015, 0.005],
        }
    )
    assert baseline_accuracies(test, thresholds) == {"majority": 0.75, "persistence": 0.75}


def test_evaluation_reports_the_model_next_to_its_baselines(trained, split):
    train, test, thresholds = split
    metrics = evaluate_models(trained, train, test, thresholds)

    clf = metrics["classification"]
    assert 0 <= clf["accuracy"] <= 1
    assert set(clf["baselines"]) == {"majority", "persistence"}
    assert 0 <= clf["logistic_regression_reference"] <= 1
    assert set(clf["per_ticker_accuracy"]) == {"AAA", "BBB", "CCC"}
    matrix = clf["confusion_matrix"]["rows_true_cols_predicted"]
    assert sum(sum(row) for row in matrix) == len(test)

    assert metrics["test_window"]["tickers"] == ["AAA", "BBB", "CCC"]
    assert metrics["train_window"]["last_date"] < metrics["test_window"]["first_date"]
    assert {"r2", "train_r2", "rmse", "mae"} <= set(metrics["regression"])


def test_the_model_learns_the_synthetic_volatility_regimes(trained, split):
    # The fixture data alternates calm and wild months, so beating "always guess the
    # most common class" is the minimum a working pipeline should manage.
    train, test, thresholds = split
    clf = evaluate_models(trained, train, test, thresholds)["classification"]
    assert clf["accuracy"] > clf["baselines"]["majority"]


def test_walk_forward_windows_are_consecutive_and_do_not_overlap(features):
    result = walk_forward(features, n_windows=3, test_days=30, gap_days=5)
    windows = result["windows"]
    assert len(windows) == 3
    for earlier, later in zip(windows, windows[1:]):
        assert earlier["last_date"] < later["first_date"]
    assert all(w["rows"] == 30 * 3 for w in windows)  # 30 dates x 3 tickers
    assert result["mean"]["accuracy"] == pytest.approx(sum(w["accuracy"] for w in windows) / 3)


def test_walk_forward_skips_windows_without_a_year_of_history(features):
    # ~400 dates: a 5th 60-day window would have to train on fewer than 250 dates.
    result = walk_forward(features, n_windows=5, test_days=60, gap_days=5)
    assert 0 < len(result["windows"]) < 5


def test_models_and_their_metrics_are_saved_together(trained, split):
    train, test, thresholds = split
    metrics = evaluate_models(trained, train, test, thresholds)
    version = save_models(trained, metrics)

    loaded = load_latest_models()
    assert set(loaded) == {"regressor", "classifier", "pca", "kmeans", "features"}
    assert loaded["features"] == trained["features"]

    saved = load_latest_metrics()
    assert saved["model_version"] == version
    assert saved["classification"]["accuracy"] == metrics["classification"]["accuracy"]


def test_release_copy_is_used_when_there_is_no_local_run(trained, tmp_path):
    import shutil

    from ml import config

    version = save_models(trained, {"note": "x"}, release=True)
    shutil.rmtree(config.MODELS_DIR / version)  # what a fresh clone looks like
    assert (config.MODELS_DIR / "release" / "classifier.pkl").exists()
    assert load_latest_models()["features"] == trained["features"]
    assert load_latest_metrics()["model_version"] == version


def test_no_models_at_all_is_a_clear_error():
    with pytest.raises(FileNotFoundError):
        load_latest_models()
    assert load_latest_metrics() is None
