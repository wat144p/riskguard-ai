import json
import pickle
import shutil
from datetime import datetime
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from . import config

# The model only sees backward-looking features.
FEATURES = [
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "volatility_5d", "volatility_20d", "price_vs_ma20",
]
MODEL_FILES = ["regressor", "classifier", "pca", "kmeans", "features"]
RELEASE_DIR_NAME = "release"


def train_models(df: pd.DataFrame) -> dict:
    """
    Trains the regressor (next-day return), the classifier (risk class),
    and PCA + KMeans (used by /recommend_similar).
    """
    X = df[FEATURES]
    y_reg = df["target_return_next_day"]
    y_clf = df["risk_class"]

    # Shallow trees, a low learning rate and early stopping: daily returns are mostly
    # noise, and anything deeper memorises the training set.
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    regressor.fit(X, y_reg)

    classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,  # early stopping
    )
    classifier.fit(X, y_clf)

    # Unsupervised: cluster rows by behaviour, on the same features
    pca = PCA(n_components=config.PCA_COMPONENTS)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=config.CLUSTERS_K, random_state=42, n_init=10)
    kmeans.fit(X_pca)

    return {
        "regressor": regressor,
        "classifier": classifier,
        "pca": pca,
        "kmeans": kmeans,
        "features": list(FEATURES),  # saved with the model so serving uses the same columns
    }


def save_models(models: dict, metrics: Optional[dict] = None, release: bool = False) -> str:
    """
    Saves the models (and the metrics of the run that produced them) to
    models/version_<timestamp>/. With release=True the same folder is also copied to
    models/release/, the one model folder that is committed to the repo.
    Returns the version string.
    """
    version = f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = config.MODELS_DIR / version
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    if metrics is not None:
        with open(save_dir / "metrics.json", "w") as f:
            json.dump({**metrics, "model_version": version}, f, indent=2)

    if release:
        release_dir = config.MODELS_DIR / RELEASE_DIR_NAME
        if release_dir.exists():
            shutil.rmtree(release_dir)
        shutil.copytree(save_dir, release_dir)

    print(f"Models saved to {save_dir}")
    return version


def latest_model_dir():
    """The newest local training run if there is one, otherwise the committed release."""
    if not config.MODELS_DIR.exists():
        raise FileNotFoundError("Models directory not found.")
    versions = sorted(
        d for d in config.MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("version_")
    )
    if versions:
        return versions[-1]
    release_dir = config.MODELS_DIR / RELEASE_DIR_NAME
    if release_dir.is_dir():
        return release_dir
    raise FileNotFoundError("No model versions found.")


def load_latest_models() -> dict:
    model_dir = latest_model_dir()
    print(f"Loading models from {model_dir}...")
    models = {}
    for name in MODEL_FILES:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def load_latest_metrics() -> Optional[dict]:
    """The metrics.json saved next to the model that is being served."""
    try:
        path = latest_model_dir() / "metrics.json"
    except FileNotFoundError:
        return None
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
