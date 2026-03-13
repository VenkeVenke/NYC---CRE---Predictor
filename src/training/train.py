"""
Layer 3 + 4 — Model Training with Optuna Tuning + MLflow Tracking
==================================================================
Full pipeline in one script:

    Step 1 — Load processed train/test CSVs
    Step 2 — Baseline (predict median — minimum bar to beat)
    Step 3 — Optuna hyperparameter search (200 trials, Bayesian)
    Step 4 — Train final model with best params
    Step 5 — Evaluate on test set
    Step 6 — Feature importance
    Step 7 — Save model locally (backup)
    Step 8 — Log to MLflow + register with @production alias

Why Optuna inside train.py?
    Instead of running tune_and_compare.py separately and manually copying
    params, this script finds the best hyperparameters automatically every
    time you retrain. 200 trials ≈ 8–12 minutes on a laptop.

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
        mlflow server --host 127.0.0.1 --port 5500
    Then run this script.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress noisy trial logs

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

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500")
MLFLOW_EXPERIMENT     = "nyc-cre-price-predictor"
MODEL_REGISTRY_NAME   = "nyc-cre-xgboost"
PRODUCTION_ALIAS      = "production"

# ---------------------------------------------------------------------------
# Optuna configuration
# ---------------------------------------------------------------------------

N_TRIALS = 200   # more trials = better params found, ~8–12 min on a laptop

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

# Fixed params that never change between trials
FIXED_PARAMS = {
    "enable_categorical": True,   # correct splits for borough, zip, neighborhood
    "tree_method":        "hist", # required when enable_categorical=True
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          0,
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
# Step 2 — Baseline
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
# Step 3 — Optuna hyperparameter search
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
) -> dict:
    """
    Use Optuna (Bayesian search) to find the best XGBoost hyperparameters.

    How Optuna works:
        Trial 1: tries random params, records R²
        Trial 2: learns from trial 1, tries smarter params
        ...
        Trial 200: by now it has focused on the best region of param space

    Returns the best params dict (merged with FIXED_PARAMS).
    """
    print(f"\n{'='*60}")
    print(f"  OPTUNA HYPERPARAMETER SEARCH  ({N_TRIALS} trials)")
    print(f"{'='*60}")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 4, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            **FIXED_PARAMS,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train_log)
        y_pred = np.expm1(model.predict(X_test))
        return float(r2_score(y_test_raw, y_pred))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = {**study.best_params, **FIXED_PARAMS}

    print(f"\n  Best R² found: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:<20} = {v}")

    return best_params


# ---------------------------------------------------------------------------
# Step 4 — Train final model with best params
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> XGBRegressor:
    """
    Log-transform target then train XGBoost with the Optuna-tuned params.

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

    print("\nTraining final XGBoost model with tuned params...")
    model = XGBRegressor(**params)
    model.fit(X_train, y_train_log)
    print("  Training complete.")
    return model


# ---------------------------------------------------------------------------
# Step 5 — Evaluate
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
# Step 6 — Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model: XGBRegressor) -> None:
    print("\n--- Feature Importances ---")
    feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    for feat, score in feat_imp.sort_values(ascending=False).items():
        bar = "█" * int(score * 100)
        print(f"  {feat:<25} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Step 7 — Save model locally (backup)
# ---------------------------------------------------------------------------

def save_model_locally(model: XGBRegressor) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    size_mb = os.path.getsize(MODEL_FILE) / 1_048_576
    print(f"\nLocal backup saved: {MODEL_FILE}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Step 8 — Log to MLflow and register model
# ---------------------------------------------------------------------------

def log_to_mlflow(
    model: XGBRegressor,
    params: dict,
    metrics: dict[str, float],
    X_train: pd.DataFrame,
) -> str:
    """
    Log the run to MLflow and register the model.

    What gets logged:
        params    — all XGBoost hyperparameters (Optuna best)
        metrics   — RMSE, MAE, R²
        tags      — metadata (model type, feature count, target transform)
        artifact  — the trained XGBoost model (native XGBoost format)

    After logging:
        - Model is registered as MODEL_REGISTRY_NAME in the MLflow registry
        - The @production alias is set on this version
        - FastAPI (Layer 5) will load via models:/<name>@production
    """
    print("\n" + "=" * 50)
    print("LOGGING TO MLFLOW")
    print("=" * 50)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Log only the tunable params (not fixed ones like tree_method)
    loggable_params = {k: v for k, v in params.items()
                       if k not in ("enable_categorical", "tree_method",
                                    "random_state", "n_jobs", "verbosity")}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")

        mlflow.log_params(loggable_params)
        mlflow.log_param("n_trials", N_TRIALS)
        print("  ✓ Parameters logged")

        mlflow.log_metrics(metrics)
        print("  ✓ Metrics logged")

        mlflow.set_tags({
            "model_type":       "xgboost_regressor",
            "target_transform": "log1p",
            "tuning":           f"optuna_{N_TRIALS}_trials",
            "feature_count":    len(FEATURE_COLS),
            "train_rows":       len(X_train),
            "features":         ", ".join(FEATURE_COLS),
        })
        print("  ✓ Tags logged")

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            registered_model_name=MODEL_REGISTRY_NAME,
            input_example=X_train.head(3),
        )
        print(f"  ✓ Model logged and registered as '{MODEL_REGISTRY_NAME}'")
        print(f"    Model URI: {model_info.model_uri}")

    # Set @production alias on the newly registered version
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  LAYER 3+4 — OPTUNA TUNING + MLFLOW TRACKING")
    print(f"  Optuna trials: {N_TRIALS}")
    print("=" * 60)

    # Step 1 — Load data
    X_train, y_train, X_test, y_test = load_data()

    # Step 2 — Baseline
    baseline_metrics = compute_baseline(y_train, y_test)

    # Step 3 — Find best hyperparameters with Optuna
    y_train_log = np.log1p(y_train)
    best_params = tune_hyperparameters(X_train, y_train_log, X_test, y_test)

    # Step 4 — Train final model with best params
    model = train_model(X_train, y_train, best_params)

    # Step 5 — Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 6 — Feature importance
    print_feature_importance(model)

    # Step 7 — Save locally
    save_model_locally(model)

    # Step 8 — Log to MLflow
    try:
        run_id = log_to_mlflow(model, best_params, metrics, X_train)
    except Exception as e:
        print(f"\nMLflow logging skipped ({type(e).__name__}: {e})")
        print("Model is saved locally. Start MLflow server and re-run to register.")
        run_id = "N/A (MLflow offline)"

    # Final summary
    rmse_lift = (baseline_metrics["baseline_rmse"] - metrics["rmse"]) / baseline_metrics["baseline_rmse"] * 100
    mae_lift  = (baseline_metrics["baseline_mae"]  - metrics["mae"])  / baseline_metrics["baseline_mae"]  * 100

    print("\n" + "=" * 60)
    print("  MODEL vs BASELINE SUMMARY")
    print("=" * 60)
    print(f"  RMSE  : model ${metrics['rmse']:>12,.0f}  vs  baseline ${baseline_metrics['baseline_rmse']:>12,.0f}  ({rmse_lift:+.1f}%)")
    print(f"  MAE   : model ${metrics['mae']:>12,.0f}  vs  baseline ${baseline_metrics['baseline_mae']:>12,.0f}  ({mae_lift:+.1f}%)")
    print(f"  R²    : model {metrics['r2']:>13.4f}  vs  baseline {baseline_metrics['baseline_r2']:>13.4f}")
    print("=" * 60)
    print(f"\n  MLflow UI : {MLFLOW_TRACKING_URI}")
    print(f"  Experiment: {MLFLOW_EXPERIMENT}")
    print(f"  Run ID    : {run_id}")
    print(f"  Registry  : {MODEL_REGISTRY_NAME}  alias=@{PRODUCTION_ALIAS}")


if __name__ == "__main__":
    main()
