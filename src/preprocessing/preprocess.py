"""
Layer 2 — Preprocessing & Feature Engineering
==============================================
Takes the raw NYC sales CSV (248,081 rows, all property types, all strings)
and produces two clean, model-ready files:

    data/processed/train.csv  — 2022–2023 clean commercial sales (train)
    data/processed/test.csv   — 2024 clean commercial sales (test)

Steps (in order):
    1. Filter to Tax Class 4 (commercial properties only)
    2. Fix data types  (all columns arrived as strings from Socrata)
    3. Filter out non-real sales (price = $0, price < $10,000)
    4. Drop rows with missing/zero gross_square_feet (can't impute building size)
    5. Fill remaining nulls  (year_built → median, land_sqft → 0)
    6. Remove outliers  (bottom 1% and top 1% of sale_price)
    7. Engineer new features  (building_age, has_commercial_units, building_class_code,
                               floor_area_ratio, log_gross_sqft, total_units, is_manhattan, sale_month)
    8. Filter extreme price_per_sqft values  ($50–$5,000/sqft)
    9. Encode neighborhood names → integer codes, save mapping to models/
    10. Select final feature columns
    11. Time-based split: 2022–2023 → train, 2024 → test
    12. Save to data/processed/
"""

import json
import os
from typing import cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_FILE      = os.path.join(BASE_DIR, "data", "raw",       "nyc_rolling_sales_raw.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_FILE    = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE     = os.path.join(PROCESSED_DIR, "test.csv")
MODELS_DIR                 = os.path.join(BASE_DIR, "models")
NEIGHBORHOOD_ENCODING_FILE = os.path.join(MODELS_DIR, "neighborhood_encoding.json")

# ---------------------------------------------------------------------------
# Step 1 — Load raw data and filter to commercial (Tax Class 4)
# ---------------------------------------------------------------------------

def load_and_filter_commercial(path: str) -> pd.DataFrame:
    """
    Load raw CSV and keep only Tax Class 4 rows.

    Tax class breakdown:
        1 = Small residential (1–3 family homes)
        2 = Large residential (apartments, co-ops, condos)
        3 = Utilities
        4 = Commercial (offices, retail, hotels, warehouses) ← we want this
    """
    print("Loading raw data...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # tax_class_at_time_of_sale is stored as int64 after CSV load
    df_comm = cast(pd.DataFrame, df[df["tax_class_at_time_of_sale"] == 4].copy())
    print(f"  After Tax Class 4 filter: {len(df_comm):,} rows")

    return df_comm


# ---------------------------------------------------------------------------
# Step 2 — Fix data types
# ---------------------------------------------------------------------------

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all relevant columns from strings to proper numeric/datetime types.

    Why are they strings? The Socrata API returns JSON — every value is a string.
    Pandas couldn't infer types because of commas in numbers ("2,300"),
    so everything stayed as object (string).
    """
    print("\nFixing data types...")

    # --- sale_price ---
    # Remove commas (e.g. "1,500,000" → "1500000") then convert to int
    df["sale_price"] = pd.to_numeric(
        df["sale_price"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    # --- gross_square_feet ---
    df["gross_square_feet"] = pd.to_numeric(
        df["gross_square_feet"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    # --- land_square_feet ---
    df["land_square_feet"] = pd.to_numeric(
        df["land_square_feet"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    # --- year_built ---
    # Already float64 from CSV (NaN forces float) — just ensure numeric
    df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")

    # --- commercial_units / residential_units ---
    df["commercial_units"]  = pd.to_numeric(df["commercial_units"],  errors="coerce")
    df["residential_units"] = pd.to_numeric(df["residential_units"], errors="coerce")

    # --- sale_date → extract sale_year and sale_month ---
    df["sale_date"]  = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_year"]  = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month

    # --- borough → already int, just make sure ---
    df["borough"] = pd.to_numeric(df["borough"], errors="coerce")

    # --- zip_code → strip any trailing ".0", convert to int
    # zip_code gives ~170 location buckets vs. only 5 boroughs — much more granular
    df["zip_code"] = pd.to_numeric(
        df["zip_code"].astype(str).str.replace(r"\.0$", "", regex=True),
        errors="coerce"
    ).fillna(0).astype(int)

    print("  Done. Key column dtypes:")
    for col in ["sale_price", "gross_square_feet", "land_square_feet",
                "year_built", "commercial_units", "residential_units",
                "sale_year", "borough", "zip_code"]:
        print(f"    {col}: {df[col].dtype}")

    return df


# ---------------------------------------------------------------------------
# Step 3 — Filter out non-real sales
# ---------------------------------------------------------------------------

def filter_real_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that don't represent real arm's-length market transactions.

    $0 sales = inter-family transfers, estate transfers, government acquisitions.
    These are legally recorded but don't reflect market value.
    We use $10,000 as the minimum to catch near-zero nominal sales too.
    """
    print("\nFiltering non-real sales...")
    before = len(df)

    df = cast(pd.DataFrame, df[df["sale_price"] > 10_000].copy())

    print(f"  Removed {before - len(df):,} rows with sale_price ≤ $10,000")
    print(f"  Remaining: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Drop rows missing gross_square_feet
# ---------------------------------------------------------------------------

def drop_missing_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where gross_square_feet is null or zero.

    Gross square footage is our most important predictor of price — we can't
    guess a building's size. Imputing it would introduce too much noise.
    This is a deliberate choice to prioritize data quality over row count.
    """
    print("\nDropping rows with missing/zero gross_square_feet...")
    before = len(df)

    df = cast(
        pd.DataFrame,
        df[df["gross_square_feet"].notna() & (df["gross_square_feet"] > 0)].copy(),
    )

    print(f"  Removed {before - len(df):,} rows")
    print(f"  Remaining: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Fill remaining nulls
# ---------------------------------------------------------------------------

def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle remaining missing values with sensible defaults.

    year_built  → fill with median year (a reasonable middle estimate)
    land_sqft   → fill with 0 (some commercial properties are condos
                  with no separate land parcel — 0 is meaningful here)
    units       → fill with 0
    """
    print("\nFilling remaining nulls...")

    year_median = df["year_built"].median()
    df["year_built"] = df["year_built"].fillna(year_median)
    print(f"  year_built nulls filled with median: {year_median:.0f}")

    df["land_square_feet"]  = df["land_square_feet"].fillna(0)
    df["commercial_units"]  = df["commercial_units"].fillna(0)
    df["residential_units"] = df["residential_units"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Step 6 — Remove outliers
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the bottom 1% and top 1% of sale_price.

    Why? A handful of extreme sales (e.g. a $5B Manhattan skyscraper or a
    $15,000 storage unit) would dominate the model's loss function and pull
    predictions away from the typical commercial property.

    1% on each end is a standard conservative cutoff — aggressive enough to
    remove true outliers, conservative enough to keep real edge cases.
    """
    print("\nRemoving outliers (bottom 1% and top 1% of sale_price)...")
    before = len(df)

    low  = df["sale_price"].quantile(0.01)
    high = df["sale_price"].quantile(0.99)
    df   = cast(
        pd.DataFrame,
        df[(df["sale_price"] >= low) & (df["sale_price"] <= high)].copy(),
    )

    print(f"  Price range after trimming: ${low:,.0f} – ${high:,.0f}")
    print(f"  Removed {before - len(df):,} outlier rows")
    print(f"  Remaining: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Feature engineering
# ---------------------------------------------------------------------------

# Mapping from raw building_class_category to simplified group
# This reduces 27 noisy categories down to 8 meaningful ones
BUILDING_CLASS_MAP = {
    "21 OFFICE BUILDINGS":                        "OFFICE",
    "43 CONDO OFFICE BUILDINGS":                  "OFFICE",
    "22 STORE BUILDINGS":                         "RETAIL",
    "46 CONDO STORE BUILDINGS":                   "RETAIL",
    "28 COMMERCIAL CONDOS":                       "RETAIL",
    "29 COMMERCIAL GARAGES":                      "GARAGE_PARKING",
    "44 CONDO PARKING":                           "GARAGE_PARKING",
    "30 WAREHOUSES":                              "WAREHOUSE_INDUSTRIAL",
    "27 FACTORIES":                               "WAREHOUSE_INDUSTRIAL",
    "49 CONDO WAREHOUSES/FACTORY/INDUS":          "WAREHOUSE_INDUSTRIAL",
    "25 LUXURY HOTELS":                           "HOTEL",
    "26 OTHER HOTELS":                            "HOTEL",
    "45 CONDO HOTELS":                            "HOTEL",
    "47 CONDO NON-BUSINESS STORAGE":              "STORAGE",
    "31 COMMERCIAL VACANT LAND":                  "VACANT_LAND",
}

# Encode simplified building class as integer for XGBoost
BUILDING_CLASS_ENCODING = {
    "OFFICE":                 0,
    "RETAIL":                 1,
    "GARAGE_PARKING":         2,
    "WAREHOUSE_INDUSTRIAL":   3,
    "HOTEL":                  4,
    "STORAGE":                5,
    "VACANT_LAND":            6,
    "OTHER":                  7,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new informative columns from existing raw columns.

    New features:
        building_age         — how old is the building at time of sale
        has_commercial_units — binary flag: does it have commercial tenants
        building_class_code  — simplified, encoded building type (int)

    We also compute price_per_sqft for analysis but DO NOT include it as
    a model feature — it directly divides the target (sale_price) so it
    would leak information to the model.
    """
    print("\nEngineering features...")

    # building_age: how many years old was the building when sold
    df["building_age"] = df["sale_year"] - df["year_built"].astype(int)
    # Cap negative values (data entry errors where year_built > sale_year)
    df["building_age"] = df["building_age"].clip(lower=0)

    # has_commercial_units: binary flag
    df["has_commercial_units"] = (df["commercial_units"] > 0).astype(int)

    # building_class_simplified: map raw category → group
    # .map() returns NaN for any category NOT in the dict — fillna catches those as "OTHER"
    # This is why .map() is better than .replace() here:
    #   .replace() only swaps exact matches and leaves everything else as-is (still strings)
    #   .map() returns NaN for unmatched values, so .fillna("OTHER") handles them all
    df["building_class_simplified"] = (
        df["building_class_category"]
        .map(BUILDING_CLASS_MAP)
        .fillna("OTHER")
    )
    # Encode simplified group → integer code, then cast to int
    df["building_class_code"] = (
        df["building_class_simplified"]
        .map(BUILDING_CLASS_ENCODING)
        .fillna(7)   # 7 = OTHER for any group not in the encoding dict
        .astype(int)
    )

    # price_per_sqft: for analysis only (not a model feature — divides the target)
    df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"]

    # -----------------------------------------------------------------------
    # Derived features (safe to use — all computed from property characteristics,
    # no information from the target sale_price)
    # -----------------------------------------------------------------------

    # floor_area_ratio: gross_sqft divided by land_sqft
    # High FAR = tall dense building (office tower). Low FAR = warehouse/garage.
    # This is a fundamental NYC zoning metric and a key CRE valuation driver.
    # +1 in denominator avoids division by zero for condo units with land_sqft=0.
    df["floor_area_ratio"] = df["gross_square_feet"] / (df["land_square_feet"] + 1)
    df["floor_area_ratio"] = df["floor_area_ratio"].clip(upper=100)  # cap extreme values

    # log_gross_sqft: explicit log of building size
    # gross_sqft spans 50 → 500,000 (a 10,000x range). Providing the log helps
    # the model distinguish small vs large with fewer tree splits.
    df["log_gross_sqft"] = np.log1p(df["gross_square_feet"])

    # total_units: combined unit count — proxy for building complexity
    df["total_units"] = df["commercial_units"] + df["residential_units"]

    # is_manhattan: Manhattan commands a premium disproportionate to other boroughs
    # Even though borough is already a feature, this explicit binary flag reinforces
    # the signal for models that may not pick it up cleanly from the raw code.
    df["is_manhattan"] = (df["borough"] == 1).astype(int)

    # sale_month: NYC CRE has strong seasonality (Q4 peaks, Q1 slowest)
    # Already extracted from sale_date in fix_dtypes()
    # (kept here as a reminder that sale_month is now in the feature set)

    print("  New features created:")
    print(f"    building_age          min={df['building_age'].min():.0f}  max={df['building_age'].max():.0f}  mean={df['building_age'].mean():.1f}")
    print(f"    has_commercial_units  values: {df['has_commercial_units'].value_counts().to_dict()}")
    print(f"    building_class_code   distribution:\n{df['building_class_simplified'].value_counts().to_string()}")
    print(f"    price_per_sqft        median=${df['price_per_sqft'].median():,.0f}")
    print(f"    floor_area_ratio      median={df['floor_area_ratio'].median():.2f}  max={df['floor_area_ratio'].max():.2f}")
    print(f"    log_gross_sqft        min={df['log_gross_sqft'].min():.2f}  max={df['log_gross_sqft'].max():.2f}")
    print(f"    total_units           median={df['total_units'].median():.0f}")
    print(f"    is_manhattan          {df['is_manhattan'].value_counts().to_dict()}")
    print(f"    sale_month            range={df['sale_month'].min():.0f}–{df['sale_month'].max():.0f}")

    return df


# ---------------------------------------------------------------------------
# Step 8 — Filter extreme price_per_sqft values
# ---------------------------------------------------------------------------

def filter_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove properties with unrealistic price per square foot.

    Why do this after the 1% outlier removal?
        The 1% trim removes extreme total prices, but a tiny cheap building
        can still have absurd $/sqft (e.g. $5 sale / 1 sqft = $5/sqft).
        Filtering on $/sqft catches a different class of bad data.

    NYC CRE $/sqft range:
        Below $50  → likely data errors or non-market transfers we missed
        Above $5,000 → ultra-premium trophy assets that skew the model
        $50–$5,000 covers the vast majority of real commercial transactions.
    """
    print("\nFiltering extreme price_per_sqft values...")
    before = len(df)

    low, high = 50, 5_000
    df = cast(
        pd.DataFrame,
        df[(df["price_per_sqft"] >= low) & (df["price_per_sqft"] <= high)].copy(),
    )

    print(f"  Kept range  : ${low}–${high:,}/sqft")
    print(f"  Removed     : {before - len(df):,} rows")
    print(f"  Remaining   : {len(df):,} rows")
    print(f"  Median $/sqft: ${df['price_per_sqft'].median():,.0f}")
    return df


# ---------------------------------------------------------------------------
# Step 9 — Encode neighborhood
# ---------------------------------------------------------------------------

def encode_neighborhood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode the neighborhood column into a numeric code.

    Why label encoding and not one-hot?
        238 unique neighborhoods → 238 extra binary columns with one-hot.
        XGBoost with enable_categorical=True handles integer-encoded categories
        correctly — no ordinal relationship is assumed between the codes.

    Encoding is sorted alphabetically so codes are stable across re-runs.
    The mapping is saved to models/neighborhood_encoding.json so the API
    can convert a neighborhood name string → code at prediction time.
    """
    print("\nEncoding neighborhoods...")

    # Normalise: strip whitespace and uppercase for consistency
    df["neighborhood"] = df["neighborhood"].str.strip().str.upper()

    # Build encoding sorted alphabetically — reproducible across runs
    unique_neighborhoods = sorted(df["neighborhood"].unique())
    encoding = {name: i for i, name in enumerate(unique_neighborhoods)}

    df["neighborhood_code"] = df["neighborhood"].map(encoding).fillna(-1).astype(int)

    print(f"  {len(encoding)} unique neighborhoods encoded (0–{len(encoding) - 1})")
    print(f"  Sample mappings: { {k: v for k, v in list(encoding.items())[:4]} }")

    return df, encoding


def save_neighborhood_encoding(encoding: dict) -> None:
    """Save neighborhood → code mapping as JSON for the API to use at inference."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(NEIGHBORHOOD_ENCODING_FILE, "w") as f:
        json.dump(encoding, f, indent=2, sort_keys=True)
    print(f"\nNeighborhood encoding saved → {NEIGHBORHOOD_ENCODING_FILE}")
    print(f"  ({len(encoding)} neighborhoods)")


# ---------------------------------------------------------------------------
# Step 10 — Select final columns
# ---------------------------------------------------------------------------

# These are the columns the model will train on (features) + target
FEATURE_COLS = [
    "borough",              # 1–5 (Manhattan to Staten Island) — treated as category
    "zip_code",             # postal code — ~170 buckets — treated as category
    "gross_square_feet",    # total building area — strongest predictor
    "land_square_feet",     # land area (0 for condo units)
    "building_age",         # years old at time of sale (replaces year_built — less redundancy)
    "commercial_units",     # number of commercial units
    "residential_units",    # number of residential units
    "has_commercial_units", # binary flag
    "building_class_code",  # encoded building type (0–7) — treated as category
    "neighborhood_code",    # encoded neighborhood (0–237) — treated as category
    "sale_year",            # year of sale (captures market trends over time)
    "sale_month",           # month of sale (1–12) — seasonal pricing signal
    "floor_area_ratio",     # gross_sqft / land_sqft — building density (core NYC metric)
    "log_gross_sqft",       # log1p(gross_sqft) — explicit size scale for the model
    "total_units",          # commercial + residential units — building complexity
    "is_manhattan",         # binary: borough == 1 — Manhattan premium signal
]

TARGET_COL = "sale_price"


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the feature columns and target.
    Drop address, neighborhood, raw text columns — the model can't use them.
    """
    cols = FEATURE_COLS + [TARGET_COL]
    df_model = cast(pd.DataFrame, df[cols].copy())

    # Final null check — should be zero at this point
    nulls = df_model.isnull().sum().sum()
    if nulls > 0:
        print(f"\n  WARNING: {nulls} nulls remain after processing:")
        print(df_model.isnull().sum()[df_model.isnull().sum() > 0])
        df_model = df_model.dropna()
        print(f"  Dropped rows with remaining nulls. Final: {len(df_model):,} rows")
    else:
        print(f"\n  No nulls remaining. Final dataset: {len(df_model):,} rows × {len(cols)} columns")

    return df_model


# ---------------------------------------------------------------------------
# Step 9 — Train / test split
# ---------------------------------------------------------------------------

def split_and_save(df: pd.DataFrame) -> None:
    """
    Time-based split: 2022–2023 sales → train, 2024 sales → test.

    Why time-based instead of random 80/20?
        In production, models always train on historical data and predict on
        future data. A random split would let 2024 prices "leak" into training,
        making test metrics look artificially better than real-world performance.

        Time-based split gives an honest measure: "how well does this model
        predict 2024 prices when only trained on 2022–2023 data?"
    """
    print("\nTime-based split: 2022–2023 → train, 2024 → test...")

    train = cast(pd.DataFrame, df[df["sale_year"] < 2024].copy())
    test  = cast(pd.DataFrame, df[df["sale_year"] == 2024].copy())

    print(f"  Train: {len(train):,} rows (2022–2023)")
    print(f"  Test:  {len(test):,} rows (2024)")

    if len(test) == 0:
        raise ValueError("No 2024 data found. Check the sale_year column.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE,   index=False)

    print(f"\n  Saved: {TRAIN_FILE}")
    print(f"  Saved: {TEST_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LAYER 2 — PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    df = load_and_filter_commercial(RAW_FILE)
    df = fix_dtypes(df)
    df = filter_real_sales(df)
    df = drop_missing_sqft(df)
    df = fill_nulls(df)
    df = remove_outliers(df)
    df = engineer_features(df)                          # computes price_per_sqft
    df = filter_price_per_sqft_outliers(df)             # NEW: filter on $/sqft
    df, neighborhood_encoding = encode_neighborhood(df) # NEW: neighborhood → code
    save_neighborhood_encoding(neighborhood_encoding)   # NEW: save JSON mapping
    df = select_columns(df)
    split_and_save(df)                                  # NEW: time-based split

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFeatures used for training: {FEATURE_COLS}")
    print(f"Target: {TARGET_COL}")


if __name__ == "__main__":
    main()
