"""Fetch NWS hourly forecast and append to data/raw/weather.csv"""
import requests
import pandas as pd
from pathlib import Path

from config import LAT, LON, DATA_RAW, USER_AGENT

OUTPUT_PATH = DATA_RAW / "weather.csv"
HEADERS = {"User-Agent": USER_AGENT}


def get_forecast_hourly_url(lat: float, lon: float) -> str:
    url = f"https://api.weather.gov/points/{lat},{lon}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()["properties"]["forecastHourly"]


def fetch_weather(forecast_url: str) -> pd.DataFrame:
    r = requests.get(forecast_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    periods = r.json()["properties"]["periods"]

    rows = []
    for p in periods:
        rows.append({
            "t": pd.to_datetime(p["startTime"]).tz_convert("UTC"),
            "temperature_f": p["temperature"],
            "wind_speed_raw": p["windSpeed"],
            "wind_direction": p["windDirection"],
            "short_forecast": p["shortForecast"],
            "is_daytime": p["isDaytime"],
        })

    df = pd.DataFrame(rows)
    df["wind_speed_mph"] = (
        df["wind_speed_raw"]
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )
    return df.drop(columns=["wind_speed_raw"])


def main():
    print("Fetching NWS grid info...")
    forecast_url = get_forecast_hourly_url(LAT, LON)
    print(f"Fetching hourly forecast: {forecast_url}")
    new_data = fetch_weather(forecast_url)

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
    print(f"Weather saved: {len(new_data)} new rows, {len(combined)} total → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
