# Bridge Clearance Dashboard

A small Flask app that displays a live, smoothed tidal chart for the
UNCW-02 gauge.

**Data flow:**

```
NOAA-Data-Pipeline-V2 (private)        do-not-hit-the-bridge (public)      this app (Render)
   data/raw/measured.csv    ──mirror──▶   data/measured.csv    ──fetch──▶   live dashboard
                          (scheduled Action, mirror-workflow/)   (raw.githubusercontent.com,
                                                                   anonymous, no token needed)
```

`NOAA-Data-Pipeline-V2` stays private. A scheduled GitHub Action living in
**do-not-hit-the-bridge** (see `mirror-workflow/` in this project — copy
it into that repo) pulls `data/raw/measured.csv` from the private repo
using a read-only token and commits it into `do-not-hit-the-bridge` as
`data/measured.csv`. This app then fetches that file straight from
`raw.githubusercontent.com` on every request (with a short server-side
cache) — no GitHub token needed here at all, since it only ever touches
a public repo.

## How it works

- `tide_data.py` downloads the CSV, auto-detects the timestamp/value
  columns, keeps the last `LOOKBACK_DAYS` of readings, and applies a
  Savitzky–Golay filter to smooth the curve for display.
- `app.py` exposes `/api/data` (JSON) and serves the dashboard page.
- `static/app.js` polls `/api/data` and redraws the Plotly chart —
  no page reload needed.
- Bridge clearance math isn't wired up yet; the chart currently shows raw
  water level. Once you have a fixed datum/offset (bridge deck height
  minus gauge datum), that subtraction can be added in `tide_data.py`
  and surfaced as its own "clearance" field.

## Configuration

Set these as environment variables (defaults shown), both locally and on
Render:

| Variable | Default | Purpose |
|---|---|---|
| `GITHUB_OWNER` | `richardmhuse` | GitHub username that owns `do-not-hit-the-bridge` |
| `DATA_REPO` | `do-not-hit-the-bridge` | Public mirror repo containing `measured.csv` |
| `DATA_BRANCH` | `main` | Branch to read from |
| `DATA_PATH` | `data/measured.csv` | Path within the mirror repo |
| `CACHE_TTL_SECONDS` | `300` | How long fetched CSV data is cached server-side |
| `LOOKBACK_DAYS` | `7` | How much history to show |
| `REFRESH_MS` | `60000` | How often the browser polls `/api/data` |

These defaults already point at `do-not-hit-the-bridge`, so nothing needs
to change for the standard setup — just make sure the mirror workflow
(below) is running so `data/measured.csv` actually exists there.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GITHUB_OWNER=your-username
python app.py
# visit http://localhost:5000
```

## Deploying to Render

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint**, point it at this repo. Render
   will read `render.yaml` and create the web service automatically.
   (Or create a Web Service manually with build command
   `pip install -r requirements.txt` and start command
   `gunicorn app:app --bind 0.0.0.0:$PORT`.)
3. Set `GITHUB_OWNER` in the Render service's environment variables.
4. Deploy. The dashboard fetches live data on every page load/poll, so
   no redeploys are needed when `measured.csv` updates upstream.

## Setting up the mirror (do-not-hit-the-bridge)

The mirror workflow lives in **`mirror-workflow/`** in this project — it's
meant to be copied into the `do-not-hit-the-bridge` repo, not this one.
Full setup steps are in `mirror-workflow/README.md`, short version:

1. Copy `mirror-workflow/.github/workflows/mirror.yml` into
   `do-not-hit-the-bridge/.github/workflows/mirror.yml`.
2. Create a fine-grained GitHub PAT scoped to **only**
   `NOAA-Data-Pipeline-V2`, with **read-only Contents** permission.
3. Add it as a repo secret named `SOURCE_REPO_TOKEN` on
   `do-not-hit-the-bridge` (Settings → Secrets and variables → Actions).
4. Run the workflow once manually (Actions tab → "Mirror measured.csv
   from NOAA-Data-Pipeline-V2" → Run workflow) to confirm `data/measured.csv`
   shows up in `do-not-hit-the-bridge`.

After that it runs on its own schedule (every 15 minutes) and this app
just reads whatever's currently there.
