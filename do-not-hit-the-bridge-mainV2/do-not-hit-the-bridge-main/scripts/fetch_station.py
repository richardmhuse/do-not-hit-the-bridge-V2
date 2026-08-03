"""
Fetch recent rainfall observations from the nearest NWS station
that actually reports precipitation.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

from config import LAT, LON, DATA_RAW, USER_AGENT, RAIN_LOOKBACK_HOURS

OUTPUT_PATH = DATA_RAW / "rain.csv"
HEADERS = {"User-Agent": USER_AGENT}


def get_observation_stations(lat: float, lon: float) -> list[dict]:
    """Return list of station feature dicts for the NWS grid point."""
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    r = requests.get(points_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    stations_url = r.json()["properties"]["observationStations"]

    r = requests.get(stations_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()["features"]


def station_reports_precip(station_url: str, lookback_hours: int = 48) -> bool:
    """
    Check recent observations; return True if any non-null precipitation
    value is present (indicates the station has a working rain gauge).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
    }
    try:
        r = requests.get(
            f"{station_url}/observations",
            headers=HEADERS,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception:
        return False

    for feat in features:
        props = feat.get("properties", {})
        for key in ("precipitationLastHour", "precipitationLast3Hours", "precipitationLast6Hours"):
            precip = props.get(key)
            if precip and precip.get("value") is not None:
                return True
    return False


def choose_rain_station(lat: float, lon: float) -> str:
    """
    Walk the ordered list of nearby stations and pick the first one
    that has recently reported precipitation.
    Falls back to the first station if none report precip.
    """
    stations = get_observation_stations(lat, lon)
    print(f"Checking {len(stations)} nearby stations for rain gauges...")

    for i, feat in enumerate(stations):
        station_url = feat["id"]          # e.g. https://api.weather.gov/stations/KILM
        station_id = station_url.rstrip("/").split("/")[-1]
        has_rain = station_reports_precip(station_url, RAIN_LOOKBACK_HOURS)
        print(f"  [{i+1}] {station_id}: {'HAS rain data' if has_rain else 'no precip'}")
        if has_rain:
            print(f"→ Selected {station_id}")
            return station_url

    # Fallback
    fallback = stations[0]["id"]
    print(f"No station with recent precip found – falling back to {fallback}")
    return fallback


def pull_rainfall(station_url: str) -> pd.DataFrame:
    """Pull recent observations and extract hourly rainfall (inches)."""
    # Request last ~7 days so we have a usable series
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 500,
    }
    r = requests.get(
        f"{station_url}/observations",
        headers=HEADERS,
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    features = r.json().get("features", [])

    rows = []
    for obs in features:
        props = obs["properties"]
        precip = props.get("precipitationLastHour")
        rainfall_mm = precip["value"] if precip and precip.get("value") is not None else None

        rows.append({
            "t": props["timestamp"],
            "rain_inches": (rainfall_mm / 25.4) if rainfall_mm is not None else 0.0,
            "station": station_url.rstrip("/").split("/")[-1],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    return df


def main():
    print("Selecting nearest station with a rain gauge...")
    station_url = choose_rain_station(LAT, LON)

    print(f"Pulling rainfall from {station_url}")
    rain_df = pull_rainfall(station_url)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not rain_df.empty:
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=["t"])
        if existing["t"].dt.tz is None:
            existing["t"] = pd.to_datetime(existing["t"], utc=True)
        combined = (
            pd.concat([existing, rain_df])
            .drop_duplicates(subset=["t"])
            .sort_values("t")
            .reset_index(drop=True)
        )
    else:
        combined = rain_df

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Rainfall saved: {len(rain_df)} new rows, {len(combined)} total → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
