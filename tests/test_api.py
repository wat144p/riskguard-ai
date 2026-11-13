"""API tests against a real (small) model trained on synthetic prices. Nothing is mocked
except the price download."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import services
from app.dependencies import get_models
from app.main import app
from ml.evaluation import evaluate_models
from ml.models import save_models

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_model_cache():
    get_models.cache_clear()
    yield
    get_models.cache_clear()


@pytest.fixture
def served(trained, split, prices, monkeypatch):
    """A saved model, and a price source that knows the three synthetic tickers."""
    train, test, thresholds = split
    save_models(trained, evaluate_models(trained, train, test, thresholds))

    def fake_fetch(tickers, use_cache=True):
        return prices[prices["ticker"].isin(tickers)].copy()

    monkeypatch.setattr(services, "fetch_stock_data", fake_fetch)
    monkeypatch.setattr(services, "TICKERS", ["AAA", "BBB", "CCC"])


def test_home_redirects_to_the_dashboard():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/static/index.html"


def test_predict_risk_returns_a_class_and_probabilities(served):
    response = client.post("/predict_risk", json={"ticker": "AAA"})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAA"
    assert body["risk_class"] in {"Low", "Medium", "High"}
    assert set(body["probabilities"]) == {"Low", "Medium", "High"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["confidence_score"] == pytest.approx(max(body["probabilities"].values()))


def test_predict_return_returns_a_number(served):
    response = client.post("/predict_return", json={"ticker": "BBB"})
    assert response.status_code == 200
    assert isinstance(response.json()["predicted_next_day_return"], float)


def test_recommend_similar_never_returns_the_input_ticker(served):
    response = client.get("/recommend_similar", params={"ticker": "AAA"})
    assert response.status_code == 200
    body = response.json()
    assert body["input_ticker"] == "AAA"
    assert all(rec["ticker"] in {"BBB", "CCC"} for rec in body["recommendations"])


def test_an_unknown_ticker_is_a_400_not_a_500(served):
    response = client.post("/predict_risk", json={"ticker": "NOPE"})
    assert response.status_code == 400
    assert "NOPE" in response.json()["detail"]


def test_a_missing_ticker_field_is_rejected(served):
    assert client.post("/predict_risk", json={}).status_code == 422


def test_without_a_model_the_api_answers_503():
    assert client.post("/predict_risk", json={"ticker": "AAA"}).status_code == 503
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_serves_the_saved_evaluation(served):
    assert client.get("/health").json()["model_loaded"] is True
    body = client.get("/metrics").json()
    assert "baselines" in body["classification"]
    assert body["model_version"].startswith("version_")
