# config.py
from pathlib import Path

# --- Location (Whiskey Creek area) ---
LAT = 34.1598087
LON = -77.8676866

# NOAA CO-OPS tide station (Wrightsville Beach)
TIDE_STATION_ID = "8658163"

# Paths
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

# Shared HTTP header (NWS requires a descriptive User-Agent)
USER_AGENT = "whiskey-creek-tide-tracker (contact@example.com)"

# How far back to look when checking stations for rain capability
RAIN_LOOKBACK_HOURS = 48
