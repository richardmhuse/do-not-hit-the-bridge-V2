"""
Compute lunar phase and illumination for a range of timestamps
and write data/raw/lunar.csv
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem
import pandas as pd

from config import DATA_RAW, LAT, LON

OUTPUT_PATH = DATA_RAW / "lunar.csv"


def lunar_features(dt: datetime) -> dict:
    """Return phase angle (0–360°), illumination fraction, and phase name."""
    observer = ephem.Observer()
    observer.lat = str(LAT)
    observer.lon = str(LON)
    observer.date = dt

    moon = ephem.Moon(observer)
    # phase = percent illuminated (0–100)
    illumination = moon.phase / 100.0
    # elongation / phase angle approximation
    phase_angle = float(moon.phase)  # ephem gives 0–100; we also expose raw

    # Simple phase name
    if illumination < 0.03:
        name = "new"
    elif illumination < 0.35:
        name = "waxing_crescent" if moon.phase < 50 else "waning_crescent"
    elif illumination < 0.65:
        name = "first_quarter" if moon.phase < 50 else "last_quarter"
    elif illumination < 0.97:
        name = "waxing_gibbous" if moon.phase < 50 else "waning_gibbous"
    else:
        name = "full"

    return {
        "illumination": round(illumination, 4),
        "phase_angle": round(float(moon.phase), 2),  # 0–100
        "phase_name": name,
    }


def main(days_back: int = 14, days_forward: int = 3, step_hours: int = 1):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days_back)
    end = now + timedelta(days=days_forward)

    rows = []
    t = start
    while t <= end:
        feats = lunar_features(t)
        rows.append({"t": t, **feats})
        t += timedelta(hours=step_hours)

    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=["t"])
        if existing["t"].dt.tz is None:
            existing["t"] = pd.to_datetime(existing["t"], utc=True)
        combined = (
            pd.concat([existing, df])
            .drop_duplicates(subset=["t"])
            .sort_values("t")
            .reset_index(drop=True)
        )
    else:
        combined = df

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Lunar data: {len(df)} new rows, {len(combined)} total → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
