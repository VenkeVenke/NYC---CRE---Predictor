"""
Layer 3 + 4 — Model Training with MLflow Tracking
===================================================
Loads processed train/test CSVs, trains XGBoost, evaluates it,
logs everything to MLflow, and registers the model.

MLflow concepts used here:

    Experiment   — a named group of related runs (ours: "nyc-cre-price-predictor")
    Run          — one execution of training, with logged params/metrics/artifacts
    Artifact     — files attached to a run (our model, feature list, etc.)
    Model Registry — versioned store of registered models
    Alias        — a human-readable pointer to a specific model version
                   e.g. @production → version 3

Why aliases instead of stages?
    MLflow 3.x replaced the old Staging/Production stages with aliases.
    Aliases are more flexible — you can have @production, @shadow, @latest, etc.
    FastAPI will load whichever version has the @production alias.

Tracking server:
    Start it first in a separate terminal:
        mlflow server --host 127.0.0.1 --port 5000
    Then run this script.
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
TRAIN_FILE  = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_FILE   = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_FILE  = os.path.join(MODELS_DIR, "xgboost_model.joblib")

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI   = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT     = "nyc-cre-price-predictor"
MODEL_REGISTRY_NAME   = "nyc-cre-xgboost"
PRODUCTION_ALIAS      = "production"

# ---------------------------------------------------------------------------
# Feature columns (must match exactly what Layer 2 produced)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "borough",              # category: 1–5
    "zip_code",             # category: ~170 NYC zip codes
    "gross_square_feet",    # continuous: total building area
    "land_square_feet",     # continuous: land area
    "building_age",         # continuous: years old at sale (replaces year_built)
    "commercial_units",     # continuous: number of commercial units
    "residential_units",    # continuous: number of residential units
    "has_commercial_units", # binary flag
    "building_class_code",  # category: 0–7 building types
    "neighborhood_code",    # category: 0–237 NYC neighborhoods
    "sale_year",            # continuous: year of sale
    "sale_month",           # continuous: month of sale (1–12) — seasonality
    "floor_area_ratio",     # continuous: gross_sqft / land_sqft — building density
    "log_gross_sqft",       # continuous: log1p(gross_sqft) — explicit size scale
    "total_units",          # continuous: commercial + residential units
    "is_manhattan",         # binary: borough == 1 — Manhattan premium
]

# Columns treated as categories — XGBoost will NOT assume numeric ordering
CATEGORICAL_COLS = ["borough", "zip_code", "building_class_code", "neighborhood_code"]

TARGET_COL = "sale_price"

# XGBoost hyperparameters
PARAMS = {
    "n_estimators":       500,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "enable_categorical": True,   # correct splits for borough, zip, neighborhood
    "tree_method":        "hist", # required when enable_categorical=True
    "random_state":       42,
    "n_jobs":             -1,
}


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def apply_categorical_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CATEGORICAL_COLS to pandas 'category' dtype.

    Why this matters:
        CSV loading reads all integers as int64. XGBoost would treat
        zip_code=10001 as a number smaller than zip_code=10002.
        Setting dtype='category' tells XGBoost to treat each value as
        its own bucket — no ordering assumed. This is critical for zip_code
        and neighborhood_code where numeric order is meaningless.
    """
    X = X.copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print("Loading processed data...")
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    print(f"  Train: {len(train):,} rows  (2022–2023)")
    print(f"  Test:  {len(test):,} rows  (2024)")

    X_train = apply_categorical_dtypes(train[FEATURE_COLS])
    y_train = train[TARGET_COL]
    X_test  = apply_categorical_dtypes(test[FEATURE_COLS])
    y_test  = test[TARGET_COL]

    print(f"  Categorical columns: {CATEGORICAL_COLS}")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Step 2 — Train model
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Log-transform target then train XGBoost.

    Why log1p?
        sale_price is heavily right-skewed ($195K–$160M — 800x range).
        Log scale compresses that into a near-normal ~12–19 range.
        XGBoost learns it much more accurately.
        Predictions are reversed with np.expm1() after inference.
    """
    print("\nApplying log transform to sale_price...")
    y_train_log = np.log1p(y_train)
    print(f"  Original  : ${y_train.min():,.0f} – ${y_train.max():,.0f}")
    print(f"  Log-scaled: {y_train_log.min():.2f} – {y_train_log.max():.2f}")

    print("\nTraining XGBoost model...")
    model = XGBRegressor(**PARAMS)
    model.fit(X_train, y_train_log)
    print("  Training complete.")
    return model


# ---------------------------------------------------------------------------
# Step 3 — Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Predict on test set (reversing log transform) and compute metrics.

    RMSE — penalises large errors heavily (in dollars)
    MAE  — plain average dollar error per prediction
    R²   — how much of price variance the model explains (0–1, higher = better)
    """
    print("\nEvaluating on test set...")
    y_pred = np.expm1(model.predict(X_test))

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    avg_price = float(y_test.mean())
    mae_pct   = mae / avg_price * 100

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  RMSE : ${rmse:>15,.0f}")
    print(f"  MAE  : ${mae:>15,.0f}  ({mae_pct:.1f}% of avg price)")
    print(f"  R²   : {r2:>16.4f}")
    print("=" * 50)

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Step 4 — Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model: XGBRegressor) -> None:
    print("\n--- Feature Importances ---")
    feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    for feat, score in feat_imp.sort_values(ascending=False).items():
        bar = "█" * int(score * 100)
        print(f"  {feat:<25} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Step 5 — Save model locally (backup)
# ---------------------------------------------------------------------------

def save_model_locally(model: XGBRegressor) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    size_mb = os.path.getsize(MODEL_FILE) / 1_048_576
    print(f"\nLocal backup saved: {MODEL_FILE}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Step 6 — Log to MLflow and register model
# ---------------------------------------------------------------------------

def log_to_mlflow(
    model: XGBRegressor,
    metrics: dict[str, float],
    X_train: pd.DataFrame,
) -> str:
    """
    Log the run to MLflow and register the model.

    What gets logged:
        params    — all XGBoost hyperparameters
        metrics   — RMSE, MAE, R²
        tags      — metadata (model type, feature count, target transform)
        artifact  — the trained XGBoost model (native XGBoost format)

    After logging:
        - Model is registered as MODEL_REGISTRY_NAME in the MLflow registry
        - The @production alias is set on this version
        - FastAPI (Layer 5) will load via models:/<name>@production

    Returns:
        run_id — the unique ID for this MLflow run
    """
    print("\n" + "=" * 50)
    print("LOGGING TO MLFLOW")
    print("=" * 50)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")

        # --- Log hyperparameters ---
        mlflow.log_params(PARAMS)
        print("  ✓ Parameters logged")

        # --- Log metrics ---
        mlflow.log_metrics(metrics)
        print("  ✓ Metrics logged")

        # --- Log useful tags ---
        mlflow.set_tags({
            "model_type":       "xgboost_regressor",
            "target_transform": "log1p",        # important: tells consumers to apply expm1
            "feature_count":    len(FEATURE_COLS),
            "train_rows":       len(X_train),
            "features":         ", ".join(FEATURE_COLS),
        })
        print("  ✓ Tags logged")

        # --- Log model to MLflow (XGBoost native format) and register it ---
        # mlflow.xgboost.log_model stores the model in MLflow's artifact store
        # registered_model_name creates an entry in the Model Registry automatically
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",                     # MLflow 3.x: use name instead of artifact_path
            registered_model_name=MODEL_REGISTRY_NAME,
            input_example=X_train.head(3),   # sample input for schema inference
        )
        print(f"  ✓ Model logged and registered as '{MODEL_REGISTRY_NAME}'")
        print(f"    Model URI: {model_info.model_uri}")

    # --- Set @production alias on the newly registered version ---
    # We do this AFTER the run closes so the version is fully registered
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get the latest version of our registered model (MLflow 3.x compatible)
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    latest_version = max(int(v.version) for v in versions)

    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias=PRODUCTION_ALIAS,
        version=latest_version,
    )
    print(f"  ✓ Alias '@{PRODUCTION_ALIAS}' → version {latest_version}")
    print(f"\n  Load this model later with:")
    print(f"    mlflow.xgboost.load_model('models:/{MODEL_REGISTRY_NAME}@{PRODUCTION_ALIAS}')")

    return run_id


# ---------------------------------------------------------------------------
# Step 6 — Baseline comparison
# ---------------------------------------------------------------------------

def compute_baseline(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """
    Dumb baseline: predict the median training sale_price for every property.

    This is the minimum bar the model must beat. If XGBoost can't outperform
    'just guess the median', then our features are useless or something is broken.

    R² of the baseline is always 0 or negative — the model should be much higher.
    """
    median_pred = float(y_train.median())
    preds = np.full(len(y_test), median_pred)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    print("\n" + "=" * 50)
    print("BASELINE (predict training median for everything)")
    print("=" * 50)
    print(f"  Median prediction : ${median_pred:,.0f}")
    print(f"  RMSE : ${rmse:>15,.0f}")
    print(f"  MAE  : ${mae:>15,.0f}")
    print(f"  R²   : {r2:>16.4f}")
    print("=" * 50)

    return {"baseline_rmse": rmse, "baseline_mae": mae, "baseline_r2": r2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LAYER 3+4 — MODEL TRAINING + MLFLOW TRACKING")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_data()

    # Step 0: baseline — sets the minimum bar for the model to beat
    baseline_metrics = compute_baseline(y_train, y_test)

    model   = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print_feature_importance(model)
    save_model_locally(model)

    try:
        run_id = log_to_mlflow(model, metrics, X_train)
    except Exception as e:
        print(f"\nMLflow logging skipped ({type(e).__name__}: {e})")
        print("Model is saved locally. Start MLflow server and re-run to register.")
        run_id = "N/A (MLflow offline)"

    # Summary: how much did we beat the baseline?
    rmse_lift = (baseline_metrics["baseline_rmse"] - metrics["rmse"]) / baseline_metrics["baseline_rmse"] * 100
    mae_lift  = (baseline_metrics["baseline_mae"]  - metrics["mae"])  / baseline_metrics["baseline_mae"]  * 100

    print("\n" + "=" * 60)
    print("MODEL vs BASELINE SUMMARY")
    print("=" * 60)
    print(f"  RMSE  : model ${metrics['rmse']:>12,.0f}  vs  baseline ${baseline_metrics['baseline_rmse']:>12,.0f}  ({rmse_lift:+.1f}%)")
    print(f"  MAE   : model ${metrics['mae']:>12,.0f}  vs  baseline ${baseline_metrics['baseline_mae']:>12,.0f}  ({mae_lift:+.1f}%)")
    print(f"  R²    : model {metrics['r2']:>13.4f}  vs  baseline {baseline_metrics['baseline_r2']:>13.4f}")
    print("=" * 60)
    print(f"\nMLflow UI : http://127.0.0.1:5000")
    print(f"Experiment: {MLFLOW_EXPERIMENT}")
    print(f"Run ID    : {run_id}")
    print(f"Registry  : {MODEL_REGISTRY_NAME}  alias=@{PRODUCTION_ALIAS}")



if __name__ == "__main__":
    main()
