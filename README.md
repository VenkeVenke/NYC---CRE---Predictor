# NYC CRE Price Predictor

End-to-end ML pipeline that predicts NYC commercial real estate sale prices.

**Stack:** Python · Pandas · XGBoost · MLflow · FastAPI · Docker · GitHub Actions · Render

---

## Project Goal

Build a production-grade machine learning pipeline that:
- Ingests real NYC property sales data (2022–2024)
- Cleans and engineers features from raw data
- Trains an XGBoost regression model to predict commercial sale prices
- Tracks experiments with MLflow
- Serves predictions via a FastAPI REST endpoint
- Runs in Docker and deploys to Render with CI/CD

---

## Project Structure

```
nyc-cre-predictor/
├── data/
│   ├── raw/            # Downloaded CSVs — gitignored, re-run ingestion to regenerate
│   └── processed/      # Cleaned + engineered data — gitignored
├── src/
│   ├── ingestion/      # Layer 1: download_data.py
│   ├── preprocessing/  # Layer 2: preprocess.py
│   ├── training/       # Layer 3: train.py
│   └── serving/        # Layer 5: app.py (FastAPI)
├── models/             # Saved model artifacts — gitignored
├── tests/              # Unit tests (Layer 7)
├── requirements.txt    # Pinned dependencies
├── .gitignore
└── README.md
```

---

## Setup (Run Once)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Build Layers

---

### ✅ Layer 1 — Data Ingestion

**What it does:**
Downloads NYC property sales data (2022–2024) from NYC Open Data's Socrata API,
loads it into a Pandas DataFrame, explores the structure, and saves it locally as a raw CSV.

**Data source:**
- Dataset: NYC Citywide Annualized Calendar Sales Update
- Socrata ID: `w2pb-icbu`
- URL: https://data.cityofnewyork.us/City-Government/NYC-Citywide-Annualized-Calendar-Sales-Update/w2pb-icbu
- Coverage: All 5 boroughs · 2022–2024

**Why Socrata API instead of per-borough Excel files?**
One endpoint, all boroughs, all years. We filter by date server-side using SoQL.
More maintainable than downloading and parsing 15 separate Excel files.

**How pagination works:**
Socrata returns max 50,000 rows per request. We use `$limit` and `$offset`
to walk through all pages until a page returns fewer rows than the page size.

**Raw data facts (as downloaded):**
- 248,081 rows × 28 columns
- All columns arrive as `object` (string) type — fixed in Layer 2
- Covers all property types (residential + commercial) — filtered in Layer 2
- Contains $0 sales (non-arm's-length transfers) — filtered in Layer 2
- File size: ~46.6 MB

**Key columns:**
| Column | Description |
|---|---|
| `borough` | 1=Manhattan 2=Bronx 3=Brooklyn 4=Queens 5=Staten Island |
| `neighborhood` | Neighborhood name |
| `building_class_category` | Broad building type (e.g. "21 OFFICE BUILDINGS") |
| `tax_class_at_time_of_sale` | 1=Small residential 2=Large residential 4=Commercial |
| `gross_square_feet` | Total building area — key feature for price prediction |
| `land_square_feet` | Land area |
| `year_built` | Year the building was constructed |
| `sale_price` | **Target variable** — what we predict |
| `sale_date` | Date the sale closed |

**What needs fixing (addressed in Layer 2):**
- All numeric columns stored as strings → need type conversion
- `sale_price = 0` rows (~many) → non-real transactions, must be removed
- `gross_square_feet` null in 45.6% of rows → drop those rows
- `year_built` null in 6.6% of rows → fill with median
- Dataset includes all tax classes → filter to Tax Class 4 (commercial only)

**Run it:**
```bash
python src/ingestion/download_data.py
```

**Output:** `data/raw/nyc_rolling_sales_raw.csv`

---

### ✅ Layer 2 — Preprocessing & Feature Engineering

**What it does:**
Takes the raw 248,081-row dataset and produces clean, model-ready train/test CSVs
by filtering, fixing types, removing bad data, engineering features, and splitting.

**Steps (in order):**

| Step | What | Why |
|---|---|---|
| 1. Filter Tax Class 4 | 248K → 16,410 rows | We only predict commercial properties |
| 2. Fix data types | Strings → int/float/datetime | Socrata API returns everything as strings |
| 3. Filter real sales | Remove price ≤ $10,000 | $0 sales are family transfers, not market prices |
| 4. Drop missing sqft | Remove null/zero gross_square_feet | Can't predict price without building size |
| 5. Fill nulls | year_built → median (1931), land_sqft → 0 | Prevent model errors on remaining nulls |
| 6. Remove outliers | Bottom 1% and top 1% of sale_price | Extreme values skew the model |
| 7. Feature engineering | Create building_age, has_commercial_units, building_class_code | More signal for the model |
| 8. Select columns | Keep 10 features + 1 target | Drop address, text, identifiers — model can't use them |
| 9. Train/test split | 80% train, 20% test, random_state=42 | Reproducible split for fair evaluation |

**Data funnel:**
```
248,081  raw rows
 16,410  after Tax Class 4 filter
 11,239  after removing $0 / near-zero sales
  4,062  after dropping missing gross_square_feet
  3,981  after outlier removal
  3,184  → train.csv
    797  → test.csv
```

**Features used for training:**
| Feature | Description |
|---|---|
| `borough` | 1–5 (Manhattan to Staten Island) |
| `gross_square_feet` | Total building area — strongest price predictor |
| `land_square_feet` | Land area (0 for condo units) |
| `year_built` | Raw construction year |
| `building_age` | `sale_year - year_built` (engineered) |
| `commercial_units` | Number of commercial units |
| `residential_units` | Number of residential units |
| `has_commercial_units` | Binary: 1 if commercial_units > 0 (engineered) |
| `building_class_code` | Encoded building type 0–7 (engineered) |
| `sale_year` | Year of sale — captures market trends |

**Target:** `sale_price`

**Building class encoding:**
```
0 = OFFICE              (21 Office Buildings, 43 Condo Office)
1 = RETAIL              (22 Store Buildings, 46 Condo Store, 28 Commercial Condos)
2 = GARAGE_PARKING      (29 Commercial Garages, 44 Condo Parking)
3 = WAREHOUSE_INDUSTRIAL(30 Warehouses, 27 Factories, 49 Condo Warehouse)
4 = HOTEL               (25 Luxury Hotels, 26 Other Hotels, 45 Condo Hotels)
5 = STORAGE             (47 Condo Non-Business Storage)
6 = VACANT_LAND         (31 Commercial Vacant Land)
7 = OTHER               (Religious, Educational, Hospital, etc.)
```

**Key dataset facts after processing:**
- Price range: $195,610 – $160,000,000
- Average building age: 78.7 years
- Median price per sqft: $507
- Most common type: Retail (1,557 rows)

**Run it:**
```bash
python src/preprocessing/preprocess.py
```

**Output:** `data/processed/train.csv` and `data/processed/test.csv`

---

### ✅ Layer 3 — Model Training

**What it does:**
Loads the processed train/test CSVs, trains an XGBoost regression model,
evaluates performance on the test set, prints feature importances,
and saves the model artifact to `models/xgboost_model.joblib`.

**Why XGBoost?**
- Best-in-class for tabular (row/column) data
- No feature scaling needed
- Handles mixed numeric + encoded categorical features well
- Fast training on CPU
- Industry standard for structured regression problems

**Model hyperparameters:**
| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 500 | Number of trees in the boosting chain |
| `max_depth` | 6 | Max depth per tree — XGBoost default, works well for most tabular data |
| `learning_rate` | 0.05 | How much each tree corrects errors — lower + more trees = better fit |
| `subsample` | 0.8 | Each tree sees 80% of rows — reduces overfitting |
| `colsample_bytree` | 0.8 | Each tree sees 80% of features — reduces overfitting |
| `random_state` | 42 | Reproducible results |

**Baseline metrics (first run):**
| Metric | Value | Meaning |
|---|---|---|
| RMSE | $10,813,613 | Avg error, large errors penalized — in dollars |
| MAE | $5,175,148 | Avg absolute error per prediction |
| R² | 0.3976 | Model explains 40% of price variance |
| MAE % of avg price | 66.5% | Typical prediction is off by ~67% |

**Why R²=0.40 is expected as a baseline:**
- Price range is $195K–$160M (800x spread) — very wide target
- Only 3,184 training rows — small dataset for CRE
- No neighborhood-level features yet
- This is the floor to improve from in future iterations

**Feature importances:**
```
gross_square_feet    0.1956  — size is the strongest price driver
building_class_code  0.1335  — office vs retail vs warehouse matters
land_square_feet     0.1156  — land value is significant
year_built           0.1130  — building age matters
commercial_units     0.1020
building_age         0.0941
borough              0.0821
sale_year            0.0719
residential_units    0.0558
has_commercial_units 0.0365
```

**Error that was caught and fixed:**
`building_class_code` was stored as `object` type because `.replace()` leaves
unrecognized categories as raw strings. Fixed by switching to `.map()` in
`preprocess.py` — `.map()` returns `NaN` for unmatched values so `.fillna(7)`
correctly assigns them to the OTHER category.

**Run it:**
```bash
python src/training/train.py
```

**Output:** `models/xgboost_model.joblib` (1.73 MB)

---

### ✅ Layer 4 — MLflow Tracking

**What it does:**
Wraps the training script with MLflow logging. Every run is recorded with
parameters, metrics, tags, and the model artifact. The model is registered
in the MLflow Model Registry and tagged with the `@production` alias.

**MLflow concepts:**
| Concept | What it is |
|---|---|
| Experiment | Named group of runs (`nyc-cre-price-predictor`) |
| Run | One training execution — logs params, metrics, artifacts |
| Artifact | Files attached to a run (the trained model) |
| Model Registry | Versioned store of registered models |
| Alias | Human-readable pointer to a model version (`@production`) |

**Why aliases instead of stages?**
MLflow 3.x replaced Staging/Production stages with aliases.
Aliases are more flexible — `@production`, `@shadow`, `@latest` etc.
FastAPI loads whichever version has the `@production` alias.

**What gets logged per run:**
- Parameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- Metrics: RMSE, MAE, R²
- Tags: model_type, target_transform (log1p), feature_count, train_rows
- Artifact: trained XGBoost model in native format

**How to load the production model:**
```python
import mlflow.xgboost
model = mlflow.xgboost.load_model("models:/nyc-cre-xgboost@production")
```

**MLflow 3.x deprecation fixes applied:**
- `artifact_path` → `name` in `log_model()`
- `get_latest_versions()` → `search_model_versions()` for registry lookup

**Run it:**
```bash
# Terminal 1 — start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Terminal 2 — run training (logs to MLflow automatically)
python src/training/train.py
```

**MLflow UI:** http://127.0.0.1:5000
- Experiments tab → see all runs, compare metrics side by side
- Models tab → see registered `nyc-cre-xgboost`, version 1, alias `@production`

---

### ✅ Layer 5 — FastAPI Serving

**What it does:**
Serves the trained model as a REST API. Loads the `@production` model from
MLflow on startup. Accepts property features via HTTP POST, returns predicted price.

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server + model status check |
| GET | `/model-info` | Which model version is currently loaded |
| POST | `/predict` | Send features → get predicted price |

**How the model loads:**
On startup, FastAPI loads from MLflow registry using the `@production` alias.
This means when Project 2 promotes a new model, FastAPI serves it automatically.

```python
model = mlflow.xgboost.load_model("models:/nyc-cre-xgboost@production")
```

**Sample request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "borough": 1, "zip_code": 10001, "gross_square_feet": 8500.0,
    "land_square_feet": 2000.0, "year_built": 1962, "building_age": 62,
    "commercial_units": 3, "residential_units": 0,
    "has_commercial_units": 1, "building_class_code": 1, "sale_year": 2024
  }'
```

**Sample response:**
```json
{
  "predicted_price": 6813619,
  "model_name": "nyc-cre-xgboost",
  "model_version": "3",
  "model_alias": "production"
}
```

**Auto-generated API docs:** http://localhost:8000/docs

**Run it:**
```bash
# Terminal 1 — MLflow server (must be running)
mlflow server --host 127.0.0.1 --port 5000

# Terminal 2 — FastAPI server
uvicorn src.serving.app:app --reload --port 8000
```

---

### ⬜ Layer 6 — Docker

*Coming after Layer 5.*

---

### ⬜ Layer 7 — GitHub Actions CI/CD

*Coming after Layer 6.*

---

### ⬜ Layer 8 — Deploy to Render

*Coming after Layer 7.*

---

## Data Notes

- Raw and processed data files are gitignored (too large, always re-generatable)
- Model artifacts are gitignored
- To regenerate raw data: run `python src/ingestion/download_data.py`
- To regenerate processed data: run `python src/preprocessing/preprocess.py` (Layer 2)
