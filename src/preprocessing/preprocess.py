"""
Layer 2 — Preprocessing & Feature Engineering
==============================================
Takes the raw NYC sales CSV (248,081 rows, all property types, all strings)
and produces two clean, model-ready files:

    data/processed/train.csv  — 80% of clean commercial sales
    data/processed/test.csv   — 20% of clean commercial sales

Steps (in order):
    1. Filter to Tax Class 4 (commercial properties only)
    2. Fix data types  (all columns arrived as strings from Socrata)
    3. Filter out non-real sales (price = $0, price < $10,000)
    4. Drop rows with missing/zero gross_square_feet (can't impute building size)
    5. Fill remaining nulls  (year_built → median, land_sqft → 0)
    6. Remove outliers  (bottom 1% and top 1% of sale_price)
    7. Engineer new features
    8. Encode building class into simplified categories
    9. Select final feature columns
    10. Train / test split (80 / 20)
    11. Save to data/processed/
"""

import os
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_FILE      = os.path.join(BASE_DIR, "data", "raw",       "nyc_rolling_sales_raw.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_FILE    = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE     = os.path.join(PROCESSED_DIR, "test.csv")

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

    # --- sale_date → extract sale_year (we use year as a feature, not raw date) ---
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_year"] = df["sale_date"].dt.year

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

    # price_per_sqft: for analysis only (not a model feature)
    df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"]

    print("  New features created:")
    print(f"    building_age          min={df['building_age'].min():.0f}  max={df['building_age'].max():.0f}  mean={df['building_age'].mean():.1f}")
    print(f"    has_commercial_units  values: {df['has_commercial_units'].value_counts().to_dict()}")
    print(f"    building_class_code   distribution:\n{df['building_class_simplified'].value_counts().to_string()}")
    print(f"    price_per_sqft        median=${df['price_per_sqft'].median():,.0f}")

    return df


# ---------------------------------------------------------------------------
# Step 8 — Select final columns
# ---------------------------------------------------------------------------

# These are the columns the model will train on (features) + target
FEATURE_COLS = [
    "borough",             # 1–5 (Manhattan to Staten Island)
    "zip_code",            # postal code — ~170 buckets, much more granular than borough
    "gross_square_feet",   # total building area — strongest predictor
    "land_square_feet",    # land area (0 for condo units)
    "year_built",          # raw year (model can use alongside building_age)
    "building_age",        # years old at time of sale
    "commercial_units",    # number of commercial units
    "residential_units",   # number of residential units
    "has_commercial_units",# binary flag
    "building_class_code", # encoded building type (0–7)
    "sale_year",           # year of sale (captures market trends over time)
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
    Split 80/20 into train and test sets. Save both as CSV.

    random_state=42 ensures the split is reproducible — every time we run
    this script we get the exact same train/test split.

    We do NOT shuffle by date here intentionally — for simplicity in Layer 2.
    A more advanced version (Layer 2+) could do a time-based split.
    """
    print("\nSplitting into train (80%) and test (20%)...")

    train, test = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.2, random_state=42),
    )

    print(f"  Train: {len(train):,} rows")
    print(f"  Test:  {len(test):,} rows")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE,  index=False)

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
    df = engineer_features(df)
    df = select_columns(df)
    split_and_save(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFeatures used for training: {FEATURE_COLS}")
    print(f"Target: {TARGET_COL}")


if __name__ == "__main__":
    main()
