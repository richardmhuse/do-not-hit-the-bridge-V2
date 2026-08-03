import logging
import os

from flask import Flask, jsonify, render_template

from tide_data import RAW_URL, fetch_tide_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

REFRESH_MS = int(os.environ.get("REFRESH_MS", "60000"))
BRIDGE_CLEARANCE_FT = float(os.environ.get("BRIDGE_CLEARANCE_FT", "4.81"))
MIN_WATER_DEPTH_FT = float(os.environ.get("MIN_WATER_DEPTH_FT", "1.86"))
INITIAL_VIEW_DAYS = float(os.environ.get("INITIAL_VIEW_DAYS", "1.5"))


@app.route("/")
def index():
    return render_template(
        "index.html",
        source_url=RAW_URL,
        refresh_ms=REFRESH_MS,
        bridge_clearance_ft=BRIDGE_CLEARANCE_FT,
        min_water_depth_ft=MIN_WATER_DEPTH_FT,
        initial_view_days=INITIAL_VIEW_DAYS,
    )


@app.route("/api/data")
def api_data():
    try:
        payload = fetch_tide_data()
        return jsonify({"ok": True, **payload})
    except Exception as exc:
        logger.exception("Failed to fetch tide data")
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
