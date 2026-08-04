"""
Pull tidal/gauge data from the NC FIMAN API and save it as a CSV table.

Source: https://fiman.nc.gov/api/gauges/<STATION_ID>
Example station used below: UNCW-02

The API returns JSON like:
{
  "historical": [
    {"value": -1.64, "at": "2026-06-22T22:10:10", "code": "00065"},
    ...
  ]
}

"code" is the USGS/NWIS parameter code. 00065 = Gauge height (ft).
"""

import requests
import pandas as pd
from pathlib import Path

# ---- Config ----------------------------------------------------------
STATION_ID = "WSC-MASE-WL"
API_URL = f"https://fiman.nc.gov/api/gauges/{STATION_ID}"
OUTPUT_PATH = Path("data/raw/measured.csv")

# Map known parameter codes to human-readable names (extend as needed)
PARAMETER_CODES = {
    "00065": "gauge_height_ft",
}


def fetch_gauge_data(url: str) -> dict:
    """Request JSON data from the FIMAN API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; tidal-data-fetch/1.0)"
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def to_dataframe(data: dict) -> pd.DataFrame:
    """Convert the FIMAN JSON payload into a tidy DataFrame."""
    records = data.get("historical", [])
    if not records:
        raise ValueError("No 'historical' records found in API response.")

    df = pd.DataFrame(records)

    # Parse timestamp
    df["at"] = pd.to_datetime(df["at"])

   # Shift tidal values up by 2
    df["value"] = df["value"] + 2

    # Friendly parameter name (falls back to raw code if unmapped)
    df["parameter"] = df["code"].map(PARAMETER_CODES).fillna(df["code"])

    # Reorder / rename columns for clarity
    df = df.rename(columns={"at": "timestamp", "value": "value"})
    df = df[["timestamp", "value", "parameter", "code"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def main():
    print(f"Fetching data for station '{STATION_ID}' from {API_URL} ...")
    data = fetch_gauge_data(API_URL)

    df = to_dataframe(data)
    print(f"Retrieved {len(df)} records "
          f"({df['timestamp'].min()} to {df['timestamp'].max()})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
