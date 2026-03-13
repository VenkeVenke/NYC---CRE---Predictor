"""
Layer 5 — FastAPI Serving
==========================
Serves the trained XGBoost model as a REST API with three endpoints:

    GET  /health       — Is the server alive?
    GET  /model-info   — Which model version is loaded?
    POST /predict      — Send property features, get back predicted price

How the model loads:
    On startup, FastAPI loads the model from MLflow registry using the
    @production alias. This means when Project 2 retraining promotes a
    new model to @production, this server picks it up automatically.

    Load URI: models:/nyc-cre-xgboost@production

Log transform:
    Our model was trained on log1p(sale_price), so predictions come back
    in log space. We reverse with np.expm1() before returning to the caller.

Run locally:
    # Make sure MLflow server is running first (from Layer 4):
    mlflow server --host 127.0.0.1 --port 5500

    # Then start this server:
    uvicorn src.serving.app:app --reload --port 8000

Test it:
    curl http://localhost:8000/health
    curl http://localhost:8000/model-info
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"borough":1,"zip_code":10001,"gross_square_feet":8500,
              "land_square_feet":2000,"building_age":62,
              "commercial_units":3,"residential_units":0,
              "has_commercial_units":1,"building_class_code":1,
              "neighborhood":"MIDTOWN WEST","sale_year":2024,"sale_month":6}'
"""

import json
import os
import socket
import numpy as np
import joblib
import mlflow.xgboost
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500")
MODEL_NAME          = os.getenv("MODEL_NAME",          "nyc-cre-xgboost")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS",         "production")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

# We store the loaded model and its metadata here so every request
# can use it without reloading from MLflow each time
model_state: dict = {
    "model":                 None,
    "version":               None,
    "alias":                 MODEL_ALIAS,
    "neighborhood_encoding": {},   # neighborhood name → integer code
}


# ---------------------------------------------------------------------------
# Startup: load model from MLflow
# ---------------------------------------------------------------------------

# Resolves relative to this file so it works both locally and in Docker.
# Local:  src/serving/../../models/  →  models/
# Docker: LOCAL_MODEL_FILE env var overrides the default → /app/models/
_HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL_MODEL_FILE = os.getenv(
    "LOCAL_MODEL_FILE",
    os.path.normpath(os.path.join(_HERE, "..", "..", "models", "xgboost_model.joblib")),
)

LOCAL_NEIGHBORHOOD_ENCODING_FILE = os.getenv(
    "NEIGHBORHOOD_ENCODING_FILE",
    os.path.normpath(os.path.join(_HERE, "..", "..", "models", "neighborhood_encoding.json")),
)


def load_production_model() -> None:
    """
    Load the @production model from MLflow registry on startup.
    Falls back to the local joblib file if MLflow is unreachable.

    Primary path (local dev):
        Connects to MLflow HTTP server, loads model via @production alias.
        This is the full production-grade path — MLflow is source of truth.

    Fallback path (Docker):
        MLflow 3.x security middleware binds to localhost-only by default,
        so the FastAPI container cannot reach the MLflow container via HTTP.
        When MLflow load fails, we fall back to the local joblib backup
        (models/xgboost_model.joblib) mounted via Docker volume.
    """
    # --- Quick connectivity check before attempting MLflow load ---
    # The MLflow SDK has no default timeout — a connection refusal can hang
    # indefinitely. We test the TCP connection first (2s timeout) to fail fast.
    def _mlflow_reachable() -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(MLFLOW_TRACKING_URI)
            host   = parsed.hostname or "127.0.0.1"
            port   = parsed.port   or 5000
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except OSError:
            return False

    # --- Try MLflow first ---
    if _mlflow_reachable():
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            print(f"Loading model from MLflow: {model_uri}")

            model_state["model"] = mlflow.xgboost.load_model(model_uri)

            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            model_state["version"] = version_info.version

            print(f"Model loaded via MLflow: {MODEL_NAME} v{model_state['version']} @{MODEL_ALIAS}")

        except Exception as e:
            print(f"MLflow reachable but load failed ({type(e).__name__}: {e})")

    # --- Fallback: load from local joblib file if model not yet loaded ---
    if model_state["model"] is None:
        print(f"MLflow unreachable or failed. Loading from: {LOCAL_MODEL_FILE}")
        model_state["model"]   = joblib.load(LOCAL_MODEL_FILE)
        model_state["version"] = "local"
        print(f"Model loaded from local file: {LOCAL_MODEL_FILE}")

    # --- Always load neighborhood encoding from JSON (needed in all paths) ---
    if os.path.exists(LOCAL_NEIGHBORHOOD_ENCODING_FILE):
        with open(LOCAL_NEIGHBORHOOD_ENCODING_FILE) as f:
            model_state["neighborhood_encoding"] = json.load(f)
        print(f"Neighborhood encoding loaded: {len(model_state['neighborhood_encoding'])} neighborhoods")
    else:
        print(f"WARNING: Neighborhood encoding not found at {LOCAL_NEIGHBORHOOD_ENCODING_FILE}")


# ---------------------------------------------------------------------------
# Lifespan: run startup logic before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    load_production_model()
    yield
    print("Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NYC CRE Price Predictor",
    description="Predicts NYC commercial real estate sale prices using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Input schema (Pydantic model)
# ---------------------------------------------------------------------------

class PropertyFeatures(BaseModel):
    """
    Input schema for a single property prediction request.

    Pydantic validates all fields automatically:
        - Wrong type (e.g. string instead of int) → 422 error, clear message
        - Missing required field → 422 error
        - Out-of-range value → 422 error (if we add validators)

    Field descriptions appear in the auto-generated API docs at /docs.
    Call GET /neighborhoods to get the full list of valid neighborhood names.
    """
    borough:              int   = Field(..., ge=1, le=5,        description="Borough code: 1=Manhattan 2=Bronx 3=Brooklyn 4=Queens 5=Staten Island")
    zip_code:             int   = Field(..., ge=10000, le=11697, description="NYC ZIP code")
    gross_square_feet:    float = Field(..., gt=0,               description="Total building area in square feet")
    land_square_feet:     float = Field(..., ge=0,               description="Land area in square feet (0 for condo units)")
    building_age:         int   = Field(..., ge=0,               description="Age of building at time of sale (sale_year - year_built)")
    commercial_units:     int   = Field(..., ge=0,               description="Number of commercial units")
    residential_units:    int   = Field(..., ge=0,               description="Number of residential units")
    has_commercial_units: int   = Field(..., ge=0, le=1,         description="Binary flag: 1 if commercial_units > 0")
    building_class_code:  int   = Field(..., ge=0, le=7,         description="Building type: 0=Office 1=Retail 2=Garage 3=Warehouse 4=Hotel 5=Storage 6=VacantLand 7=Other")
    neighborhood:         str   = Field(...,                     description="NYC neighborhood name in uppercase e.g. 'MIDTOWN WEST'. Call GET /neighborhoods for all valid values.")
    sale_year:            int   = Field(..., ge=2022, le=2030,   description="Year of sale")
    sale_month:           int   = Field(..., ge=1,   le=12,      description="Month of sale (1=Jan … 12=Dec)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "borough":              1,
                "zip_code":             10001,
                "gross_square_feet":    8500.0,
                "land_square_feet":     2000.0,
                "building_age":         62,
                "commercial_units":     3,
                "residential_units":    0,
                "has_commercial_units": 1,
                "building_class_code":  1,
                "neighborhood":         "MIDTOWN WEST",
                "sale_year":            2024,
                "sale_month":           6,
            }
        }
    }


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_price: int    = Field(..., description="Predicted sale price in USD")
    model_name:      str    = Field(..., description="Registered model name in MLflow")
    model_version:   str    = Field(..., description="Model version number from MLflow registry")
    model_alias:     str    = Field(..., description="Model alias used to load (e.g. 'production')")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the server is running and model is loaded.
    Used by Docker health checks, load balancers, and Render.
    """
    model_loaded = model_state["model"] is not None
    return {
        "status":       "ok" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_name":   MODEL_NAME,
        "model_alias":  MODEL_ALIAS,
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """
    Returns information about the currently loaded model version.
    Useful for verifying that a new model was promoted and loaded correctly.
    """
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name":    MODEL_NAME,
        "model_version": model_state["version"],
        "model_alias":   model_state["alias"],
        "mlflow_uri":    MLFLOW_TRACKING_URI,
    }


@app.get("/neighborhoods", tags=["Prediction"])
def list_neighborhoods():
    """
    Returns all valid neighborhood names accepted by the /predict endpoint.

    Use this to find the exact neighborhood string to send in your request.
    All names are uppercase (e.g. 'MIDTOWN WEST', 'FLUSHING-NORTH').
    """
    encoding = model_state["neighborhood_encoding"]
    if not encoding:
        raise HTTPException(status_code=503, detail="Neighborhood encoding not loaded.")
    return {
        "count":         len(encoding),
        "neighborhoods": sorted(encoding.keys()),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: PropertyFeatures):
    """
    Predict the sale price of a NYC commercial property.

    Send property features as a JSON body.
    Returns the predicted sale price in USD.

    The model was trained on log1p(sale_price).
    Predictions are automatically reversed with expm1() before returning.

    Call GET /neighborhoods to get the full list of valid neighborhood names.
    """
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    # Convert neighborhood name → integer code
    encoding  = model_state["neighborhood_encoding"]
    nbhd_name = features.neighborhood.strip().upper()
    if nbhd_name not in encoding:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown neighborhood '{features.neighborhood}'. Call GET /neighborhoods for valid values.",
        )
    neighborhood_code = encoding[nbhd_name]

    # Compute derived features (same logic as preprocess.py — must stay in sync)
    floor_area_ratio = min(
        features.gross_square_feet / (features.land_square_feet + 1),
        100.0,          # clip at 100 to match training
    )
    log_gross_sqft = float(np.log1p(features.gross_square_feet))
    total_units    = features.commercial_units + features.residential_units
    is_manhattan   = 1 if features.borough == 1 else 0

    # Build feature array in exact column order the model was trained on
    # Order must match FEATURE_COLS in train.py exactly
    feature_values = [
        features.borough,
        features.zip_code,
        features.gross_square_feet,
        features.land_square_feet,
        features.building_age,
        features.commercial_units,
        features.residential_units,
        features.has_commercial_units,
        features.building_class_code,
        neighborhood_code,
        features.sale_year,
        features.sale_month,
        floor_area_ratio,
        log_gross_sqft,
        total_units,
        is_manhattan,
    ]
    X = np.array([feature_values], dtype=np.float32)

    # Predict in log space, then reverse the log transform
    log_prediction = model_state["model"].predict(X)
    price_dollars  = float(np.expm1(log_prediction[0]))

    return PredictionResponse(
        predicted_price=int(round(price_dollars)),
        model_name=MODEL_NAME,
        model_version=str(model_state["version"]),
        model_alias=model_state["alias"],
    )
