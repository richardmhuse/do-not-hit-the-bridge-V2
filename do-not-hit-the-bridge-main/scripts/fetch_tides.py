"""Fetch NOAA CO-OPS tide predictions and append to data/raw/tides.csv"""
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

from config import TIDE_STATION_ID, DATA_RAW, USER_AGENT

OUTPUT_PATH = DATA_RAW / "tides.csv"


def fetch_tides(station_id: str, begin_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": begin_date,
        "end_date": end_date,
        "station": station_id,
        "product": "predictions",
        "datum": "MLLW",
        "time_zone": "gmt",          # store everything in UTC
        "interval": "h",
        "units": "english",
        "application": "whiskey-creek-tracker",
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "predictions" not in data:
        raise ValueError(f"Unexpected response: {data}")
    df = pd.DataFrame(data["predictions"])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"v": "tide_ft"})   # clearer column name
    return df[["t", "tide_ft"]]


def main():
    today = date.today()
    # Pull a little extra history so the series is continuous
    today = date.today()
    start = today - timedelta(days=30)
    end   = today + timedelta(days=3)

    print(f"Fetching tides for station {TIDE_STATION_ID} ({start} → {end})...")
    new_data = fetch_tides(
        TIDE_STATION_ID,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=["t"])
        if existing["t"].dt.tz is None:
            existing["t"] = pd.to_datetime(existing["t"], utc=True)
        combined = (
            pd.concat([existing, new_data])
            .drop_duplicates(subset=["t"])
            .sort_values("t")
            .reset_index(drop=True)
        )
    else:
        combined = new_data.sort_values("t").reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Tides saved: {len(new_data)} new rows, {len(combined)} total → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
