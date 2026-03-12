"""
Layer 7 — API Unit Tests
=========================
Tests for the three FastAPI endpoints:

    GET  /health       — server alive, model loaded
    GET  /model-info   — model metadata returned
    POST /predict      — valid input → positive price, invalid input → 422

How model loading is handled in tests:
    The real app loads a model from MLflow or a local .joblib file on startup.
    We don't want tests to depend on real files or a running MLflow server.
    So we patch load_production_model() to inject a mock model directly into
    model_state — the same dict the endpoints read from.

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared test payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "borough":              1,
    "zip_code":             10001,
    "gross_square_feet":    8500.0,
    "land_square_feet":     2000.0,
    "year_built":           1962,
    "building_age":         62,
    "commercial_units":     3,
    "residential_units":    0,
    "has_commercial_units": 1,
    "building_class_code":  1,
    "sale_year":            2024,
}


# ---------------------------------------------------------------------------
# Fixture: mock the model so tests run without real files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Creates a TestClient with a mocked model.

    The mock model's predict() returns np.array([15.5]), which in log space
    means expm1(15.5) ≈ $5.9M — a realistic Manhattan retail price.

    We use scope="module" so the app starts once and all tests share it,
    keeping the suite fast.
    """
    import src.serving.app as app_module

    # Build a mock XGBoost model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([15.5])  # log1p space → ~$5.9M

    # Replace load_production_model() with a function that sets model_state directly
    def fake_load():
        app_module.model_state["model"]   = mock_model
        app_module.model_state["version"] = "test-1"

    with patch.object(app_module, "load_production_model", side_effect=fake_load):
        with TestClient(app_module.app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# Tests: /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_model_loaded(client):
    data = client.get("/health").json()
    assert data["model_loaded"] is True


def test_health_has_model_name(client):
    data = client.get("/health").json()
    assert data["model_name"] == "nyc-cre-xgboost"


# ---------------------------------------------------------------------------
# Tests: /model-info
# ---------------------------------------------------------------------------

def test_model_info_returns_200(client):
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_has_version(client):
    data = client.get("/model-info").json()
    assert "model_version" in data


def test_model_info_has_alias(client):
    data = client.get("/model-info").json()
    assert data["model_alias"] == "production"


# ---------------------------------------------------------------------------
# Tests: /predict — valid input
# ---------------------------------------------------------------------------

def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_has_price(client):
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert "predicted_price" in data


def test_predict_price_is_positive(client):
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert data["predicted_price"] > 0


def test_predict_price_is_integer(client):
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert isinstance(data["predicted_price"], int)


def test_predict_returns_model_name(client):
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert data["model_name"] == "nyc-cre-xgboost"


# ---------------------------------------------------------------------------
# Tests: /predict — invalid input (should return 422 Unprocessable Entity)
# ---------------------------------------------------------------------------

def test_predict_invalid_borough(client):
    """Borough must be 1-5. Sending 9 should fail validation."""
    bad = {**VALID_PAYLOAD, "borough": 9}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_negative_sqft(client):
    """gross_square_feet must be > 0."""
    bad = {**VALID_PAYLOAD, "gross_square_feet": -100}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_field(client):
    """Missing required field should return 422."""
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "zip_code"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_wrong_type(client):
    """Sending string instead of int should return 422."""
    bad = {**VALID_PAYLOAD, "borough": "manhattan"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
