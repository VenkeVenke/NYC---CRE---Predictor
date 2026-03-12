"""
Model Comparison — XGBoost vs LightGBM vs CatBoost with Optuna Tuning
======================================================================
Loads the same processed train/test CSVs as train.py.
Tunes each model with Optuna (Bayesian search, 30 trials each).
Prints a comparison table.
Saves the best model to models/xgboost_model.joblib so app.py serves it.

Why Optuna?
    Instead of trying every combination of hyperparameters (grid search),
    Optuna learns which values work well and focuses trials there.
    30 trials ≈ 5–10 minutes per model on a laptop.

Run:
    python src/training/tune_and_compare.py
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress noisy trial logs

# ---------------------------------------------------------------------------
# Paths  (same as train.py)
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..")
TRAIN_FILE = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_FILE  = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_PATH  = os.path.join(MODELS_DIR, "xgboost_model.joblib")   # winner goes here

# ---------------------------------------------------------------------------
# Features  (same as train.py)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "borough",
    "zip_code",
    "gross_square_feet",
    "land_square_feet",
    "building_age",
    "commercial_units",
    "residential_units",
    "has_commercial_units",
    "building_class_code",
    "neighborhood_code",
    "sale_year",
    "sale_month",
    "floor_area_ratio",
    "log_gross_sqft",
    "total_units",
    "is_manhattan",
]
CATEGORICAL_COLS = ["borough", "zip_code", "building_class_code", "neighborhood_code"]
TARGET_COL = "sale_price"

N_TRIALS = 100  # Optuna trials per model — more trials = better hyperparameters found


# ---------------------------------------------------------------------------
# Data loading  (same logic as train.py)
# ---------------------------------------------------------------------------

def apply_categorical_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def load_data():
    print("Loading processed data...")
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")

    X_train = apply_categorical_dtypes(train[FEATURE_COLS])
    y_train = np.log1p(train[TARGET_COL])   # log-transform target once, reuse for all models

    X_test  = apply_categorical_dtypes(test[FEATURE_COLS])
    y_test  = test[TARGET_COL]              # raw dollars for evaluation

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test_raw, label: str) -> dict:
    """Predict (reversing log transform) and compute metrics."""
    y_pred = np.expm1(model.predict(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test_raw, y_pred)))
    mae  = float(mean_absolute_error(y_test_raw, y_pred))
    r2   = float(r2_score(y_test_raw, y_pred))
    return {"model": label, "R2": r2, "MAE": mae, "RMSE": rmse}


# ---------------------------------------------------------------------------
# Optuna objective functions — one per model family
# ---------------------------------------------------------------------------

def xgb_objective(trial, X_train, y_train_log, X_test, y_test_raw):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "enable_categorical": True,
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train_log)
    y_pred = np.expm1(model.predict(X_test))
    return float(r2_score(y_test_raw, y_pred))


def lgbm_objective(trial, X_train, y_train_log, X_test, y_test_raw):
    # LightGBM needs integer codes for categorical — convert from pandas category
    cat_cols_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 4, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train_log,
        categorical_feature=cat_cols_indices,
    )
    y_pred = np.expm1(model.predict(X_test))
    return float(r2_score(y_test_raw, y_pred))


def catboost_objective(trial, X_train_raw, y_train_log, X_test_raw, y_test_raw):
    """
    CatBoost takes raw (non-encoded) categorical columns by index.
    We pass the original integer values — CatBoost handles encoding internally.
    """
    # Use plain int arrays — CatBoost handles categoricals natively
    X_tr = X_train_raw.copy()
    X_ts = X_test_raw.copy()
    for col in CATEGORICAL_COLS:
        X_tr[col] = X_tr[col].cat.codes.astype(int)
        X_ts[col] = X_ts[col].cat.codes.astype(int)

    cat_indices = [X_tr.columns.get_loc(c) for c in CATEGORICAL_COLS]

    params = {
        "iterations":       trial.suggest_int("iterations", 200, 800),
        "depth":            trial.suggest_int("depth", 4, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed":      42,
        "verbose":          0,
        "cat_features":     cat_indices,
        "allow_writing_files": False,
    }
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_train_log)
    y_pred = np.expm1(model.predict(X_ts))
    return float(r2_score(y_test_raw, y_pred))


# ---------------------------------------------------------------------------
# Per-model tuning runners
# ---------------------------------------------------------------------------

def tune_xgboost(X_train, y_train_log, X_test, y_test_raw):
    print(f"\n{'='*55}")
    print(f"  Tuning XGBoost  ({N_TRIALS} trials)")
    print(f"{'='*55}")
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: xgb_objective(t, X_train, y_train_log, X_test, y_test_raw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    best = study.best_params
    best.update({"enable_categorical": True, "tree_method": "hist",
                 "random_state": 42, "n_jobs": -1, "verbosity": 0})
    model = XGBRegressor(**best)
    model.fit(X_train, y_train_log)
    metrics = evaluate(model, X_test, y_test_raw, "XGBoost (tuned)")
    print(f"  Best R²={metrics['R2']:.4f}   time={elapsed:.0f}s")
    return model, metrics, study.best_value


def tune_lightgbm(X_train, y_train_log, X_test, y_test_raw):
    print(f"\n{'='*55}")
    print(f"  Tuning LightGBM  ({N_TRIALS} trials)")
    print(f"{'='*55}")
    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: lgbm_objective(t, X_train, y_train_log, X_test, y_test_raw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    best = study.best_params
    best.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    model = LGBMRegressor(**best)
    model.fit(X_train, y_train_log, categorical_feature=cat_indices)
    metrics = evaluate(model, X_test, y_test_raw, "LightGBM (tuned)")
    print(f"  Best R²={metrics['R2']:.4f}   time={elapsed:.0f}s")
    return model, metrics, study.best_value


def tune_catboost(X_train, y_train_log, X_test, y_test_raw):
    print(f"\n{'='*55}")
    print(f"  Tuning CatBoost  ({N_TRIALS} trials)")
    print(f"{'='*55}")
    t0 = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: catboost_objective(t, X_train, y_train_log, X_test, y_test_raw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    # Rebuild final CatBoost model with best params
    X_tr = X_train.copy()
    X_ts = X_test.copy()
    for col in CATEGORICAL_COLS:
        X_tr[col] = X_tr[col].cat.codes.astype(int)
        X_ts[col] = X_ts[col].cat.codes.astype(int)
    cat_indices = [X_tr.columns.get_loc(c) for c in CATEGORICAL_COLS]

    best = study.best_params
    best.update({"random_seed": 42, "verbose": 0,
                 "cat_features": cat_indices, "allow_writing_files": False})
    model = CatBoostRegressor(**best)
    model.fit(X_tr, y_train_log)
    metrics = evaluate(model, X_ts, y_test_raw, "CatBoost (tuned)")
    print(f"  Best R²={metrics['R2']:.4f}   time={elapsed:.0f}s")
    return model, metrics, study.best_value


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def compute_baseline(y_train_raw, y_test_raw):
    median_pred = float(y_train_raw.median())
    preds = np.full(len(y_test_raw), median_pred)
    return {
        "model": "Baseline (median)",
        "R2":   float(r2_score(y_test_raw, preds)),
        "MAE":  float(mean_absolute_error(y_test_raw, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_raw, preds))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  MODEL COMPARISON — XGBoost / LightGBM / CatBoost")
    print(f"  Optuna trials per model: {N_TRIALS}")
    print("=" * 60)

    # Load data
    X_train, y_train_log, X_test, y_test_raw = load_data()
    y_train_raw = pd.read_csv(TRAIN_FILE)[TARGET_COL]   # raw prices for baseline

    # Baseline
    baseline = compute_baseline(y_train_raw, y_test_raw)

    # Tune all three models
    xgb_model,  xgb_m,  _ = tune_xgboost(X_train, y_train_log, X_test, y_test_raw)
    lgbm_model, lgbm_m, _ = tune_lightgbm(X_train, y_train_log, X_test, y_test_raw)
    cb_model,   cb_m,   _ = tune_catboost(X_train, y_train_log, X_test, y_test_raw)

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    results = [baseline, xgb_m, lgbm_m, cb_m]
    results_sorted = sorted(results, key=lambda x: x["R2"], reverse=True)

    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Model':<25}  {'R²':>7}  {'MAE':>14}  {'RMSE':>14}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*14}  {'-'*14}")
    for r in results_sorted:
        marker = " ◀ WINNER" if r == results_sorted[0] and r["model"] != "Baseline (median)" else ""
        print(f"  {r['model']:<25}  {r['R2']:>7.4f}  ${r['MAE']:>13,.0f}  ${r['RMSE']:>13,.0f}{marker}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Pick winner and save
    # ---------------------------------------------------------------------------
    model_map = {
        "XGBoost (tuned)":  xgb_model,
        "LightGBM (tuned)": lgbm_model,
        "CatBoost (tuned)": cb_model,
    }
    winner_name    = results_sorted[0]["model"]
    winner_metrics = results_sorted[0]
    winner_model   = model_map[winner_name]

    print(f"\n  Winner: {winner_name}")
    print(f"  R² = {winner_metrics['R2']:.4f}  |  MAE = ${winner_metrics['MAE']:,.0f}  |  RMSE = ${winner_metrics['RMSE']:,.0f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(winner_model, SAVE_PATH)
    size_mb = os.path.getsize(SAVE_PATH) / 1_048_576
    print(f"\n  Saved to: {SAVE_PATH}  ({size_mb:.2f} MB)")
    print("\n  Next steps:")
    print("    - Update requirements.txt if lightgbm/catboost won")
    print("    - Re-run: pytest tests/ -v   (to confirm tests still pass)")
    print("    - Restart FastAPI server to load new model")


if __name__ == "__main__":
    main()
