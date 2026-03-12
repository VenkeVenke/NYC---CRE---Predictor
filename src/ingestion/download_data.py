"""
Layer 1 — Data Ingestion
========================
Downloads NYC Citywide Annualized Calendar Sales data (2022–2024)
from the NYC Open Data Socrata API (dataset ID: w2pb-icbu).

Why Socrata API?
  - One endpoint covers all 5 boroughs + all years
  - No API key required for public datasets
  - SoQL (Socrata Query Language) lets us filter by date range server-side
    so we only download the rows we need

How pagination works:
  - Socrata returns up to 50,000 rows per request
  - We use $limit and $offset to walk through all matching rows in chunks
  - We stop when a chunk comes back empty
"""

import os
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Socrata API endpoint for the combined annualized sales dataset
BASE_URL = "https://data.cityofnewyork.us/resource/w2pb-icbu.json"

# We want sales dated 2022-01-01 through 2024-12-31
DATE_START = "2022-01-01T00:00:00.000"
DATE_END   = "2024-12-31T23:59:59.999"

# Rows per API page (Socrata max is 50,000)
PAGE_SIZE = 50_000

# Where to save the raw output
RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUT_FILE = os.path.join(RAW_DIR, "nyc_rolling_sales_raw.csv")


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def fetch_page(offset: int) -> list[dict]:
    """
    Fetch one page of results from the Socrata API.

    Parameters:
        offset: How many rows to skip before returning results.
                First call: offset=0, second: offset=50000, etc.

    Returns:
        A list of dicts (each dict = one sale row).
        Returns an empty list when there are no more rows.
    """
    params = {
        "$where": f"sale_date between '{DATE_START}' and '{DATE_END}'",
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "sale_date ASC",  # consistent ordering across pages
    }

    response = requests.get(BASE_URL, params=params, timeout=60)

    # If the API returns an error, raise it so we see the message clearly
    response.raise_for_status()

    return response.json()


def download_all_pages() -> pd.DataFrame:
    """
    Walk through all pages of the Socrata API and return a single DataFrame.
    Prints progress so we can see it working.
    """
    all_rows = []
    offset = 0
    page_num = 0

    print(f"Fetching NYC sales data: {DATE_START[:10]} to {DATE_END[:10]}")
    print(f"Page size: {PAGE_SIZE:,} rows per request\n")

    while True:
        page_num += 1
        print(f"  Page {page_num} — offset {offset:,} ...", end=" ", flush=True)

        rows = fetch_page(offset)

        if not rows:
            print("no more rows. Done.")
            break

        all_rows.extend(rows)
        print(f"got {len(rows):,} rows  (total so far: {len(all_rows):,})")

        if len(rows) < PAGE_SIZE:
            # Last page — fewer rows than the page size means we've hit the end
            break

        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite to the API

    print(f"\nTotal rows downloaded: {len(all_rows):,}")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Exploration helpers
# ---------------------------------------------------------------------------

def explore_dataframe(df: pd.DataFrame) -> None:
    """
    Print a clear summary of the raw DataFrame so we know what we're working with.
    """
    print("\n" + "=" * 60)
    print("RAW DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n--- Column Names & Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- First 3 rows ---")
    print(df.head(3).to_string())

    print("\n--- Missing values per column ---")
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df) * 100).round(1)
    null_summary = pd.DataFrame({"null_count": nulls, "null_pct": nulls_pct})
    print(null_summary[null_summary["null_count"] > 0].to_string())

    print("\n--- BOROUGH value counts ---")
    if "borough" in df.columns:
        print(df["borough"].value_counts().to_string())

    print("\n--- SALE_PRICE sample values (raw, before type conversion) ---")
    if "sale_price" in df.columns:
        print(df["sale_price"].head(10).to_string())

    print("\n--- TAX_CLASS_AT_PRESENT unique values ---")
    if "tax_class_at_present" in df.columns:
        print(df["tax_class_at_present"].value_counts().to_string())

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure the output directory exists
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download all data
    df = download_all_pages()

    # Show us what we got
    explore_dataframe(df)

    # Save to CSV
    df.to_csv(OUT_FILE, index=False)
    print(f"\nRaw data saved to: {OUT_FILE}")
    print(f"File size: {os.path.getsize(OUT_FILE) / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
