"""
Orchestrator: run all fetchers + combine in the correct order.
Designed to be the single entry-point for a Render Cron Job
or a GitHub Action step.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "scripts/fetch_tides.py",
    "scripts/fetch_weather.py",
    "scripts/fetch_rain.py",
    "scripts/fetch_lunar.py",
    # measurements – keep your existing script name
    "scripts/fetch_uncw_02.py",
    "scripts/combine_data.py",
    "scripts/build_features.py",
    "scripts/train_xgboost.py",      
    "scripts/generate_forecast.py",
]


def run(script: str) -> bool:
    path = Path(script)
    if not path.exists():
        print(f"⚠  Skipping missing script: {script}")
        return False
    print(f"\n>>> Running {script}")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"✗ {script} failed with code {result.returncode}")
        return False
    print(f"✓ {script} finished")
    return True


def main():
    print("Starting full data pipeline…")
    failures = []
    for script in SCRIPTS:
        ok = run(script)
        if not ok and "fetch_uncw" not in script and "combine" not in script:
            # non-critical fetchers can fail without stopping the whole job
            failures.append(script)

    if failures:
        print(f"\nCompleted with warnings: {failures}")
        # still exit 0 so the Cron doesn’t keep retrying forever
        sys.exit(0)
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
