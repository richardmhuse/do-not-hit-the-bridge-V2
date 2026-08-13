const REFRESH_MS = window.__REFRESH_MS__ || 60000;
const BRIDGE_CLEARANCE_FT =
  typeof window.__BRIDGE_CLEARANCE_FT__ === "number" ? window.__BRIDGE_CLEARANCE_FT__ : 4.81;
const MIN_WATER_DEPTH_FT =
  typeof window.__MIN_WATER_DEPTH_FT__ === "number" ? window.__MIN_WATER_DEPTH_FT__ : 1.86;
const WARNING_MARGIN_FT = 0.2; // tint the readout within this margin of either threshold

// On touch devices, use pinch-to-zoom + single-finger pan instead of the
// desktop rectangular drag-to-zoom (which is awkward with a finger).
const IS_TOUCH_DEVICE = "ontouchstart" in window || navigator.maxTouchPoints > 0;

// Plotly's date axis has no concept of timezones — it renders whatever
// calendar values it's given, literally, with no conversion of its own.
// The API sends true UTC ("...Z") timestamps. Without this, the whole
// chart (axis ticks, the predicted line, the "now" marker) renders in
// UTC rather than the viewer's own local time — which is what was
// making the "now" marker (and everything else) look hours off.
// Auto-detects the viewer's browser/OS timezone (DST-aware).
const DISPLAY_TIME_ZONE = Intl.DateTimeFormat().resolvedOptions().timeZone;

const INITIAL_VIEW_DAYS =
  typeof window.__INITIAL_VIEW_DAYS__ === "number" ? window.__INITIAL_VIEW_DAYS__ : 1.5;
const INITIAL_VIEW_END_PADDING_HOURS = 3; // breathing room past "now" so the marker isn't flush against the edge
const DENSE_STEP_MS = 5 * 60 * 1000;

const SHORT_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIME_ZONE,
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
});

const DATE_ONLY_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIME_ZONE,
  month: "short",
  day: "numeric",
});

const TIME_ONLY_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIME_ZONE,
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
});

function formatShortLocal(isoOrMs) {
  const d = typeof isoOrMs === "number" ? new Date(isoOrMs) : new Date(isoOrMs);
  return SHORT_TIME_FORMATTER.format(d);
}

/** Compact cycle range: "Aug 13: 6:01 AM – 11:38 AM" (same day) or
 *  "Aug 13: 10:00 PM – Aug 14: 3:15 AM" (spans midnight). */
function formatCycleRange(startMs, endMs) {
  const start = new Date(startMs);
  const end = new Date(endMs);
  const startDate = DATE_ONLY_FORMATTER.format(start);
  const endDate = DATE_ONLY_FORMATTER.format(end);
  const startTime = TIME_ONLY_FORMATTER.format(start);
  const endTime = TIME_ONLY_FORMATTER.format(end);
  if (startDate === endDate) {
    return `${startDate}: ${startTime} – ${endTime}`;
  }
  return `${startDate}: ${startTime} – ${endDate}: ${endTime}`;
}

const VIEWER_PARTS_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: DISPLAY_TIME_ZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

function toViewerPlotTimestamp(utcIso) {
  const d = new Date(utcIso);
  const parts = {};
  for (const part of VIEWER_PARTS_FORMATTER.formatToParts(d)) {
    parts[part.type] = part.value;
  }
  // hour12:false can render midnight as "24" in some engines — normalize it
  const hour = parts.hour === "24" ? "00" : parts.hour;
  return `${parts.year}-${parts.month}-${parts.day}T${hour}:${parts.minute}:${parts.second}`;
}

function toViewerPlotTimestamps(utcIsoArray) {
  return utcIsoArray.map(toViewerPlotTimestamp);
}

/* ------------------------------------------------------------------ */
/*  User threshold preferences (localStorage)                          */
/* ------------------------------------------------------------------ */
const PREFS_KEY = "whiskey-creek-thresholds";

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function savePrefs(prefs) {
  localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

function getEffectiveThresholds() {
  const prefs = loadPrefs();
  return {
    showMax: prefs?.showMax ?? true,
    showMin: prefs?.showMin ?? true,
    showCrossings: prefs?.showCrossings ?? false,
    minValue: typeof prefs?.minValue === "number" ? prefs.minValue : MIN_WATER_DEPTH_FT,
    maxValue: BRIDGE_CLEARANCE_FT, // still the fixed bridge clearance
  };
}

function buildThresholdShapesAndAnnotations(crossings) {
  const { showMax, showMin, minValue, maxValue } = getEffectiveThresholds();
  const shapes = [];
  const annotations = [];

  if (showMax) {
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: maxValue,
      y1: maxValue,
      line: {
        color: "rgba(255,179,64,.82)",
        width: 1.8,
      },
    });
    annotations.push({
      xref: "paper",
      x: 1,
      xanchor: "right",
      yref: "y",
      y: maxValue,
      yshift: 14,
      showarrow: false,
      align: "right",
      text: `<b>Bridge Clearance</b><br>${maxValue.toFixed(2)} ft`,
      font: {
        family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
        size: 12,
        color: "#C97A00",
      },
      bgcolor: "rgba(255,255,255,.75)",
      borderpad: 4,
    });
  }

  if (showMin) {
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: minValue,
      y1: minValue,
      line: {
        color: "rgba(255,105,97,.82)",
        width: 1.8,
      },
    });
    annotations.push({
      xref: "paper",
      x: 1,
      xanchor: "right",
      yref: "y",
      y: minValue,
      yshift: -14,
      showarrow: false,
      align: "right",
      text: `<b>Minimum Depth</b><br>${minValue.toFixed(2)} ft`,
      font: {
        family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
        size: 12,
        color: "#D94B43",
      },
      bgcolor: "rgba(255,255,255,.75)",
      borderpad: 4,
    });

    // Crossing markers are drawn as a single Plotly trace (see buildTraces)
    // when showCrossings is enabled — not as shapes/annotations (avoids doubles).
  }

  return { shapes, annotations };
}

/* ------------------------------------------------------------------ */
/*  Settings panel (created automatically)                             */
/* ------------------------------------------------------------------ */
function createThresholdPanel() {
  // Avoid duplicating the panel if the script is ever re-run
  if (document.getElementById("threshold-panel")) return;

  const panel = document.createElement("div");
  panel.id = "threshold-panel";
  panel.innerHTML = `
    <button id="threshold-toggle" type="button" aria-expanded="false" title="Threshold settings">
      ⚙ Thresholds
    </button>
    <div id="threshold-controls" hidden>
      <label class="th-row">
        <input type="checkbox" id="show-max" checked>
        <span>Show maximum (Bridge Clearance)</span>
      </label>
      <label class="th-row">
        <input type="checkbox" id="show-min" checked>
        <span>Show minimum threshold</span>
      </label>
      <label class="th-row">
        <input type="checkbox" id="show-crossings">
        <span>Show min-depth crossings</span>
      </label>
      <label class="th-row">
        <span>Min value (ft)</span>
        <input type="number" id="min-value" step="0.01" min="0" value="${MIN_WATER_DEPTH_FT}">
      </label>
    </div>
  `;

  // Minimal styling so it looks decent on most dashboards
  const style = document.createElement("style");
  style.textContent = `
    #threshold-panel {
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 1000;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
      font-size: 13px;
    }
    #threshold-toggle {
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 10px;
      padding: 8px 14px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      font-size: 13px;
      color: #1d1d1f;
    }
    #threshold-toggle:hover {
      background: #fff;
    }
    #threshold-controls {
      margin-top: 8px;
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(0,0,0,0.1);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.1);
      min-width: 240px;
    }
    .th-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      cursor: pointer;
      color: #1d1d1f;
    }
    .th-row:last-child {
      margin-bottom: 0;
    }
    #min-value {
      width: 72px;
      padding: 4px 6px;
      border: 1px solid rgba(0,0,0,0.15);
      border-radius: 6px;
      font-size: 13px;
    }
    #min-value:disabled {
      opacity: 0.45;
    }
  `;
  document.head.appendChild(style);
  document.body.appendChild(panel);

  // Toggle open/close
  const toggleBtn = document.getElementById("threshold-toggle");
  const controls = document.getElementById("threshold-controls");
  toggleBtn.addEventListener("click", () => {
    const isHidden = controls.hidden;
    controls.hidden = !isHidden;
    toggleBtn.setAttribute("aria-expanded", String(isHidden));
  });
}

function initThresholdControls() {
  createThresholdPanel();

  const prefs = getEffectiveThresholds();
  const showMaxEl = document.getElementById("show-max");
  const showMinEl = document.getElementById("show-min");
  const showCrossingsEl = document.getElementById("show-crossings");
  const minValueEl = document.getElementById("min-value");

  showMaxEl.checked = prefs.showMax;
  showMinEl.checked = prefs.showMin;
  if (showCrossingsEl) showCrossingsEl.checked = prefs.showCrossings;
  minValueEl.value = prefs.minValue;
  minValueEl.disabled = !prefs.showMin;

  function onChange() {
    const showMax = showMaxEl.checked;
    const showMin = showMinEl.checked;
    const showCrossings = showCrossingsEl ? showCrossingsEl.checked : false;
    let minValue = parseFloat(minValueEl.value);
    if (Number.isNaN(minValue) || minValue < 0) minValue = MIN_WATER_DEPTH_FT;

    minValueEl.disabled = !showMin;
    savePrefs({ showMax, showMin, showCrossings, minValue });
    applyThresholds();
    // Rebuild chart so crossing markers appear/disappear
    if (lastPayload && chartInitialized) {
      const traces = buildTraces(lastPayload);
      nowTraceIndex = traces.length - 1;
      const series = buildCombinedSeries(lastPayload);
      const crossings = findThresholdCrossings(series, minValue);
      const { shapes, annotations } = buildThresholdShapesAndAnnotations(crossings);
      const gd = document.getElementById("chart");
      const xRange = gd.layout?.xaxis?.range ? [...gd.layout.xaxis.range] : undefined;
      Plotly.react(
        gd,
        traces,
        {
          ...CHART_LAYOUT,
          shapes,
          annotations,
          xaxis: { ...CHART_LAYOUT.xaxis, ...(xRange ? { range: xRange } : {}) },
        },
        { displayModeBar: false, responsive: true, scrollZoom: true }
      );
      updateCycleCards(lastPayload);
    } else if (lastPayload) {
      updateCycleCards(lastPayload);
    }
  }

  showMaxEl.addEventListener("change", onChange);
  showMinEl.addEventListener("change", onChange);
  if (showCrossingsEl) showCrossingsEl.addEventListener("change", onChange);
  minValueEl.addEventListener("change", onChange);
}

function applyThresholds() {
  if (!chartInitialized || !lastPayload) return;
  const series = buildCombinedSeries(lastPayload);
  const { minValue } = getEffectiveThresholds();
  const crossings = findThresholdCrossings(series, minValue);
  const { shapes, annotations } = buildThresholdShapesAndAnnotations(crossings);
  Plotly.relayout("chart", { shapes, annotations });
}


/* ------------------------------------------------------------------ */
/*  Continuous series + threshold crossings + navigable windows        */
/* ------------------------------------------------------------------ */

function buildCombinedSeries(data) {
  const timesMs = [];
  const values = [];
  const kinds = [];
  const measuredTimes = (data.timestamps || []).map((t) => new Date(t).getTime());
  const measuredVals = data.smoothed || [];
  for (let i = 0; i < measuredTimes.length; i++) {
    if (!Number.isFinite(measuredTimes[i]) || !Number.isFinite(measuredVals[i])) continue;
    timesMs.push(measuredTimes[i]);
    values.push(measuredVals[i]);
    kinds.push("measured");
  }
  const hasPrediction =
    data.predicted_timestamps && data.predicted_values && data.predicted_timestamps.length > 1;
  if (hasPrediction) {
    const predTimes = data.predicted_timestamps.map((t) => new Date(t).getTime());
    const predVals = data.predicted_values;
    const lastMeasuredMs = timesMs.length ? timesMs[timesMs.length - 1] : predTimes[0];
    let startIdx = 0;
    while (startIdx < predTimes.length && predTimes[startIdx] <= lastMeasuredMs) startIdx++;
    let prevT = lastMeasuredMs;
    let prevV = values.length ? values[values.length - 1] : predVals[startIdx] ?? 0;
    for (let i = startIdx; i < predTimes.length; i++) {
      const t = predTimes[i];
      const v = predVals[i];
      if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
      const span = t - prevT;
      if (span > DENSE_STEP_MS) {
        const n = Math.floor(span / DENSE_STEP_MS);
        for (let k = 1; k < n; k++) {
          const frac = k / n;
          timesMs.push(prevT + span * frac);
          values.push(prevV + (v - prevV) * frac);
          kinds.push("predicted");
        }
      }
      timesMs.push(t);
      values.push(v);
      kinds.push("predicted");
      prevT = t;
      prevV = v;
    }
  }
  return { timesMs, values, kinds };
}

function findThresholdCrossings(series, threshold) {
  const { timesMs, values } = series;
  const crossings = [];
  if (timesMs.length < 2) return crossings;
  for (let i = 0; i < timesMs.length - 1; i++) {
    const y0 = values[i];
    const y1 = values[i + 1];
    if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
    const below0 = y0 < threshold;
    const below1 = y1 < threshold;
    if (below0 === below1) continue;
    const span = y1 - y0;
    const frac = span === 0 ? 0 : (threshold - y0) / span;
    const ms = timesMs[i] + (timesMs[i + 1] - timesMs[i]) * frac;
    crossings.push({
      ms,
      value: threshold,
      direction: below0 && !below1 ? "up" : "down",
    });
  }
  return crossings;
}

function findNavigableWindows(series, threshold) {
  const crossings = findThresholdCrossings(series, threshold);
  const { timesMs, values } = series;
  if (!timesMs.length) return [];
  const windows = [];
  let cursor = timesMs[0];
  let above = values[0] >= threshold;
  for (const c of crossings) {
    if (above) windows.push({ startMs: cursor, endMs: c.ms, open: true });
    above = c.direction === "up";
    cursor = c.ms;
  }
  if (above) {
    windows.push({ startMs: cursor, endMs: timesMs[timesMs.length - 1], open: true });
  }
  return windows;
}

function pickCurrentAndNextWindows(windows, nowMs) {
  let current = null;
  let next = null;
  for (const w of windows) {
    if (w.startMs <= nowMs && nowMs <= w.endMs) current = w;
    else if (w.startMs > nowMs && !next) next = w;
  }
  return { current, next };
}

function updateCycleCards(data) {
  const currentRangeEl = document.getElementById("cycle-current-range");
  const currentDetailEl = document.getElementById("cycle-current-detail");
  const nextRangeEl = document.getElementById("cycle-next-range");
  const nextDetailEl = document.getElementById("cycle-next-detail");
  const currentCard = document.getElementById("cycle-current");
  const nextCard = document.getElementById("cycle-next");
  if (!currentRangeEl || !nextRangeEl) return;

  const { minValue } = getEffectiveThresholds();
  const series = buildCombinedSeries(data);
  const windows = findNavigableWindows(series, minValue);
  const nowMs = Date.now();
  const { current, next } = pickCurrentAndNextWindows(windows, nowMs);

  for (const el of [currentCard, nextCard]) {
    if (!el) continue;
    el.classList.remove("is-open", "is-closed", "is-pending");
  }

  if (current) {
    currentCard?.classList.add("is-open");
    currentRangeEl.textContent = formatCycleRange(current.startMs, current.endMs);
    const remainingMs = Math.max(0, current.endMs - nowMs);
    const remainingHrs = remainingMs / 3600000;
    currentDetailEl.textContent =
      remainingHrs >= 1
        ? "Above " + minValue.toFixed(2) + " ft · ~" + remainingHrs.toFixed(1) + " h left"
        : "Above " + minValue.toFixed(2) + " ft · ~" + Math.round(remainingMs / 60000) + " min left";
  } else {
    currentCard?.classList.add("is-closed");
    const past = windows.filter((w) => w.endMs < nowMs);
    const lastPast = past.length ? past[past.length - 1] : null;
    if (lastPast) {
      currentRangeEl.textContent = "Ended " + formatShortLocal(lastPast.endMs);
      currentDetailEl.textContent =
        "Below " + minValue.toFixed(2) + " ft · waiting for next rise";
    } else {
      currentRangeEl.textContent = "Below minimum";
      currentDetailEl.textContent = "Level under " + minValue.toFixed(2) + " ft";
    }
  }

  if (next) {
    nextCard?.classList.add("is-pending");
    nextRangeEl.textContent = formatCycleRange(next.startMs, next.endMs);
    const untilMs = Math.max(0, next.startMs - nowMs);
    const untilHrs = untilMs / 3600000;
    nextDetailEl.textContent =
      untilHrs >= 1
        ? "Starts in ~" + untilHrs.toFixed(1) + " h · above " + minValue.toFixed(2) + " ft"
        : "Starts in ~" + Math.round(untilMs / 60000) + " min · above " + minValue.toFixed(2) + " ft";
  } else {
    nextRangeEl.textContent = "—";
    nextDetailEl.textContent = "No further window in forecast";
  }
}

/* ------------------------------------------------------------------ */
/*  Original chart plumbing (slightly adapted)                         */
/* ------------------------------------------------------------------ */
const dot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const readout = document.getElementById("readout");
const readoutMeta = document.getElementById("readout-meta");
const errorBanner = document.getElementById("error-banner");

const CHART_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: {
    l: 58,
    r: 30,
    t: 28,
    b: 54,
  },
  font: {
    family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
    size: 13,
    color: "#1d1d1f",
  },
  showlegend: false,
  hovermode: false,
  hoverlabel: {
    // Invisible labels — readout at top shows the value instead
    bgcolor: "rgba(0,0,0,0)",
    bordercolor: "rgba(0,0,0,0)",
    font: { size: 1, color: "rgba(0,0,0,0)" },
  },
  dragmode: IS_TOUCH_DEVICE ? false : "pan",
  xaxis: {
    tickformat: "%b %-d\n%I %p",
    showgrid: true,
    gridcolor: "rgba(0,0,0,.03)",
    gridwidth: 1,
    zeroline: false,
    showline: false,
    ticks: "",
    ticklen: 0,
    tickfont: {
      size: 12,
      color: "#86868b",
    },
    showspikes: false,
  },
  yaxis: {
    title: {
      text: "Water Level (ft)",
      font: {
        size: 13,
        color: "#6e6e73",
      },
    },
    showspikes: false,
    showgrid: true,
    gridcolor: "rgba(0,0,0,.045)",
    gridwidth: 1,
    zeroline: false,
    showline: false,
    ticks: "",
    ticklen: 0,
    tickfont: {
      size: 12,
      color: "#86868b",
    },
  },
  // shapes & annotations are now supplied dynamically
  shapes: [],
  annotations: [],
  transition: {
    duration: 500,
    easing: "cubic-in-out",
  },
};

let chartInitialized = false;
let lastPayload = null;
let nowTraceIndex = null;
let lastCrossings = [];
let isHoveringChart = false;
let hoverReadoutValue = null;

function computeInitialXRange(data) {
  const lastMs = Math.max(new Date(data.latest_timestamp).getTime(), Date.now());
  const endMs = lastMs + INITIAL_VIEW_END_PADDING_HOURS * 3600000;
  const startMs = endMs - INITIAL_VIEW_DAYS * 24 * 3600000;
  return [
    toViewerPlotTimestamp(new Date(startMs).toISOString()),
    toViewerPlotTimestamp(new Date(endMs).toISOString()),
  ];
}

function buildTraces(data) {
  const series = buildCombinedSeries(data);
  const { minValue } = getEffectiveThresholds();
  lastCrossings = findThresholdCrossings(series, minValue);

  const measuredX = [];
  const measuredY = [];
  const predictedX = [];
  const predictedY = [];
  for (let i = 0; i < series.timesMs.length; i++) {
    const x = toViewerPlotTimestamp(new Date(series.timesMs[i]).toISOString());
    if (series.kinds[i] === "measured") {
      measuredX.push(x);
      measuredY.push(series.values[i]);
    } else {
      predictedX.push(x);
      predictedY.push(series.values[i]);
    }
  }
  if (measuredX.length && predictedX.length) {
    predictedX.unshift(measuredX[measuredX.length - 1]);
    predictedY.unshift(measuredY[measuredY.length - 1]);
  }

  const traces = [
    {
      x: toViewerPlotTimestamps(data.timestamps || []),
      y: data.raw || [],
      mode: "markers",
      marker: { color: "rgba(124, 147, 168, 0.35)", size: 3 },
      hovertemplate:
        "<b>%{y:.2f} ft</b><br>" +
        "Measured<br>" +
        "%{x|%b %d, %I:%M %p}" +
        "<extra></extra>",
      name: "raw",
    },
    {
      x: measuredX,
      y: measuredY,
      mode: "lines",
      line: { color: "#0A84FF", width: 4, shape: "linear" },
      fill: "tozeroy",
      fillcolor: "rgba(10,132,255,.14)",
      name: "Water Level",
      hovertemplate:
        "<b>%{y:.2f} ft</b><br>" +
        "Water Level<br>" +
        "%{x|%b %d, %I:%M %p}" +
        "<extra></extra>",
      showlegend: false,
    },
  ];

  if (predictedX.length > 1) {
    traces.push({
      x: predictedX,
      y: predictedY,
      mode: "lines",
      line: { color: "#0A84FF", width: 4, dash: "dot", shape: "linear" },
      opacity: 0.55,
      name: "Predicted",
      hovertemplate:
        "<b>%{y:.2f} ft</b><br>" +
        "Predicted<br>" +
        "%{x|%b %d, %I:%M %p}" +
        "<extra></extra>",
    });
  }

  // Single set of crossing markers (no shapes/annotations behind them)
  const { showCrossings } = getEffectiveThresholds();
  if (showCrossings && lastCrossings.length) {
    const cx = lastCrossings.map((c) =>
      toViewerPlotTimestamp(new Date(c.ms).toISOString())
    );
    const cy = lastCrossings.map((c) => c.value);
    const ctext = lastCrossings.map(
      (c) => (c.direction === "up" ? "Rise " : "Fall ") + formatShortLocal(c.ms)
    );
    traces.push({
      x: cx,
      y: cy,
      mode: "markers+text",
      text: ctext,
      textposition: lastCrossings.map((c) =>
        c.direction === "up" ? "top center" : "bottom center"
      ),
      textfont: {
        family: "-apple-system",
        size: 10,
        color: lastCrossings.map((c) =>
          c.direction === "up" ? "#1B8A3E" : "#C0392B"
        ),
      },
      marker: {
        size: 11,
        color: lastCrossings.map((c) =>
          c.direction === "up" ? "#34C759" : "#FF453A"
        ),
        line: { color: "#fff", width: 2 },
        symbol: lastCrossings.map((c) =>
          c.direction === "up" ? "triangle-up" : "triangle-down"
        ),
      },
      hoverinfo: "skip",
      name: "crossings",
      showlegend: false,
    });
  }

  const seed = interpolateNowValue(data, Date.now());
  traces.push({
    x: [toViewerPlotTimestamp(seed.utcIso)],
    y: [seed.value],
    mode: "markers+text",
    text: ["Now"],
    textposition: "top center",
    textfont: { family: "-apple-system", size: 11, color: "#0A84FF" },
    marker: {
      size: 13,
      color: "#FFFFFF",
      line: { color: "#0A84FF", width: 3 },
    },
    hoverinfo: "skip",
    showlegend: false,
  });

  return traces;
}

/**
 * Finds where "now" sits along the predicted segment.
 */
function interpolateNowValue(payload, nowMs) {
  const hasPrediction = payload.predicted_timestamps && payload.predicted_timestamps.length > 1;
  if (!hasPrediction) {
    return { utcIso: payload.latest_timestamp, value: payload.latest_value };
  }
  const times = payload.predicted_timestamps.map((t) => new Date(t).getTime());
  const values = payload.predicted_values;
  if (nowMs <= times[0]) {
    return { utcIso: payload.predicted_timestamps[0], value: values[0] };
  }
  if (nowMs >= times[times.length - 1]) {
    return {
      utcIso: payload.predicted_timestamps[times.length - 1],
      value: values[times.length - 1],
    };
  }
  for (let i = 0; i < times.length - 1; i++) {
    if (nowMs >= times[i] && nowMs <= times[i + 1]) {
      const span = times[i + 1] - times[i];
      const frac = span === 0 ? 0 : (nowMs - times[i]) / span;
      return {
        utcIso: new Date(nowMs).toISOString(),
        value: values[i] + (values[i + 1] - values[i]) * frac,
      };
    }
  }
  return {
    utcIso: payload.predicted_timestamps[times.length - 1],
    value: values[times.length - 1],
  };
}

function setStatus(ok) {
  if (dot) dot.classList.toggle("stale", !ok);
  if (statusText) {
    statusText.textContent = ok
      ? "Live · Whiskey Creek"
      : "Feed unavailable — showing last known data";
  }
}

/**
 * Pinch-zoom workaround for mobile.
 */
function setupMobileChartGestures(gd) {
  // One finger on date axis → horizontal pan
  // Two fingers → pinch zoom
  // (One finger on plot body is handled by setupChartHover for scrub / page scroll)
  let pinchState = null;
  let axisPanState = null;

  function touchDistance(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  function touchMidpoint(touches) {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  function toMillis(v) {
    if (v instanceof Date) return v.getTime();
    if (typeof v === "number") return v;
    const t = new Date(v).getTime();
    return Number.isNaN(t) ? Number(v) : t;
  }

  function rangeToViewer(msArr) {
    // Keep axis range in the same viewer-local timestamp space as the traces
    return msArr.map((ms) => toViewerPlotTimestamp(new Date(ms).toISOString()));
  }

  function isOnXAxis(clientY) {
    const fullLayout = gd._fullLayout;
    if (!fullLayout || !fullLayout._size) return false;
    const rect = gd.getBoundingClientRect();
    // Plotly plot area ends at top + t + h; axis labels + margin sit below
    const plotBottom = rect.top + fullLayout._size.t + fullLayout._size.h;
    // Generous hit band: from slightly above the plot bottom through the full bottom margin
    // (date ticks are often tight; users miss a narrow strip)
    const axisTop = plotBottom - 24;
    return clientY >= axisTop && clientY <= rect.bottom + 4;
  }

  function beginPinch(e) {
    const fullLayout = gd._fullLayout;
    const xa = fullLayout && fullLayout.xaxis;
    const ya = fullLayout && fullLayout.yaxis;
    if (!xa || !ya || typeof xa.p2d !== "function") return;

    const rect = gd.getBoundingClientRect();
    const mid = touchMidpoint(e.touches);
    const localX = mid.x - rect.left - fullLayout._size.l;
    const localY = mid.y - rect.top - fullLayout._size.t;

    pinchState = {
      startDistance: touchDistance(e.touches),
      anchorDataX: toMillis(xa.p2d(localX)),
      anchorDataY: ya.p2d(localY),
      startXRange: xa.range.map(toMillis),
      startYRange: [...ya.range],
    };
    axisPanState = null;
  }

  function beginAxisPan(e) {
    const fullLayout = gd._fullLayout;
    const xa = fullLayout && fullLayout.xaxis;
    if (!xa) return;

    const span = toMillis(xa.range[1]) - toMillis(xa.range[0]);
    const plotWidthPx = Math.max(1, fullLayout._size.w);

    axisPanState = {
      startClientX: e.touches[0].clientX,
      startXRange: xa.range.map(toMillis),
      dataPerPx: span / plotWidthPx,
    };
    pinchState = null;
  }

  gd.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        e.stopPropagation();
        beginPinch(e);
        return;
      }
      if (e.touches.length === 1 && isOnXAxis(e.touches[0].clientY)) {
        e.preventDefault();
        e.stopPropagation();
        beginAxisPan(e);
        return;
      }
      // One finger on plot body → leave for scrub / page scroll
      pinchState = null;
      axisPanState = null;
    },
    { passive: false, capture: true }
  );

  gd.addEventListener(
    "touchmove",
    (e) => {
      if (e.touches.length === 2 && pinchState) {
        e.preventDefault();
        e.stopPropagation();
        const newDistance = touchDistance(e.touches);
        if (newDistance < 1) return;

        const scale = pinchState.startDistance / newDistance;
        const clampedScale = Math.min(Math.max(scale, 0.05), 20);

        const newXMillis = pinchState.startXRange.map(
          (v) => pinchState.anchorDataX + (v - pinchState.anchorDataX) * clampedScale
        );
        const midY =
          (Number(pinchState.startYRange[0]) + Number(pinchState.startYRange[1])) / 2;
        const newY = pinchState.startYRange.map(
          (v) => midY + (Number(v) - midY) * clampedScale
        );

        Plotly.relayout(gd, {
          "xaxis.range": rangeToViewer(newXMillis),
          "yaxis.range": newY,
        });
        return;
      }

      if (e.touches.length === 1 && axisPanState) {
        e.preventDefault();
        e.stopPropagation();
        const dxPx = e.touches[0].clientX - axisPanState.startClientX;
        // Drag right → look at earlier times (content moves with finger)
        const shift = -dxPx * axisPanState.dataPerPx;
        const newXMillis = axisPanState.startXRange.map((v) => v + shift);
        Plotly.relayout(gd, {
          "xaxis.range": rangeToViewer(newXMillis),
        });
      }
    },
    { passive: false, capture: true }
  );

  function endGestures(e) {
    if (!e.touches || e.touches.length < 2) pinchState = null;
    if (!e.touches || e.touches.length < 1) axisPanState = null;
  }
  gd.addEventListener("touchend", endGestures, { capture: true });
  gd.addEventListener("touchcancel", endGestures, { capture: true });
}


function formatTimestamp(iso) {
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    timeZone: DISPLAY_TIME_ZONE,
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

async function refresh() {
  try {
    const res = await fetch("/api/data");
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Unknown error");
    }

    if (errorBanner) errorBanner.classList.remove("visible");
    setStatus(true);

    lastPayload = data;
    updateReadout(data);

    if (readoutMeta) {
      readoutMeta.textContent = `Last measured reading ${data.latest_value.toFixed(2)} ft at ${formatTimestamp(data.latest_timestamp)}`;
    }

    updateCycleCards(data);

    const traces = buildTraces(data);
    nowTraceIndex = traces.length - 1;

    const { shapes, annotations } = buildThresholdShapesAndAnnotations(lastCrossings);
    const plotlyConfig = { displayModeBar: false, responsive: true, scrollZoom: true };

    if (!chartInitialized) {
      const initialLayout = {
        ...CHART_LAYOUT,
        shapes,
        annotations,
        xaxis: { ...CHART_LAYOUT.xaxis, range: computeInitialXRange(data) },
      };
      Plotly.newPlot("chart", traces, initialLayout, plotlyConfig);
      chartInitialized = true;
      const gd0 = document.getElementById("chart");
      setupChartHover(gd0);
      if (IS_TOUCH_DEVICE) {
        setupMobileChartGestures(gd0);
      }
    } else {
      const gd = document.getElementById("chart");
      const layout = {
        ...CHART_LAYOUT,
        shapes,
        annotations,
        xaxis: { ...CHART_LAYOUT.xaxis },
      };
      // Preserve whatever view the user is currently looking at
      if (gd.layout?.xaxis?.range) {
        layout.xaxis.range = [...gd.layout.xaxis.range];
      } else {
        layout.xaxis.range = computeInitialXRange(data);
      }
      Plotly.react(gd, traces, layout, plotlyConfig);
    }
  } catch (err) {
    setStatus(false);
    if (errorBanner) {
      errorBanner.textContent = `Could not load gauge data: ${err.message}`;
      errorBanner.classList.add("visible");
    }
  }
}


function setupChartHover(gd) {
  if (gd._whiskeyHoverBound) return;
  gd._whiskeyHoverBound = true;

  // Custom scrubber: blue vertical line + top readout. No Plotly hover UI (avoids black marks).
  function sampleAtClientX(clientX) {
    const fullLayout = gd._fullLayout;
    if (!fullLayout || !fullLayout.xaxis || typeof fullLayout.xaxis.p2d !== "function") return null;
    if (!lastPayload) return null;

    const rect = gd.getBoundingClientRect();
    const localX = clientX - rect.left - fullLayout._size.l;
    // Clamp to plot area
    if (localX < 0 || localX > fullLayout._size.w) return null;

    const dataX = fullLayout.xaxis.p2d(localX);
    // dataX may be a Date or date-string in plot coords (viewer-local)
    const targetMs = dataX instanceof Date ? dataX.getTime() : new Date(dataX).getTime();
    if (!Number.isFinite(targetMs)) return null;

    const series = buildCombinedSeries(lastPayload);
    if (!series.timesMs.length) return null;

    // Nearest sample in the combined series
    let best = 0;
    let bestDist = Math.abs(series.timesMs[0] - targetMs);
    for (let i = 1; i < series.timesMs.length; i++) {
      const d = Math.abs(series.timesMs[i] - targetMs);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    // Linear interpolate between neighbors for a continuous value
    let value = series.values[best];
    let ms = series.timesMs[best];
    if (best > 0 && best < series.timesMs.length - 1) {
      const left = series.timesMs[best] <= targetMs ? best : best - 1;
      const right = left + 1;
      const t0 = series.timesMs[left];
      const t1 = series.timesMs[right];
      if (t1 > t0 && targetMs >= t0 && targetMs <= t1) {
        const frac = (targetMs - t0) / (t1 - t0);
        value = series.values[left] + (series.values[right] - series.values[left]) * frac;
        ms = targetMs;
      }
    }
    return { ms, value, plotX: dataX };
  }

  function showScrub(sample) {
    if (!sample) return;
    isHoveringChart = true;
    hoverReadoutValue = sample.value;

    if (readout) {
      readout.innerHTML = `${sample.value.toFixed(2)}<span class="unit">ft</span>`;
    }
    const { minValue, maxValue } = getEffectiveThresholds();
    const nearHigh = maxValue - sample.value <= WARNING_MARGIN_FT;
    const nearLow = sample.value - minValue <= WARNING_MARGIN_FT;
    if (readout) {
      readout.classList.toggle("near-limit-high", nearHigh && !nearLow);
      readout.classList.toggle("near-limit-low", nearLow);
    }
    if (readoutMeta) {
      readoutMeta.textContent = `At ${formatHoverAxisTime(sample.ms)}`;
    }

    // Blue vertical scrub line (shape) — only visual, no black hover chrome
    const xPlot =
      typeof sample.plotX === "string"
        ? sample.plotX
        : toViewerPlotTimestamp(new Date(sample.ms).toISOString());

    const baseShapes = (gd.layout && gd.layout.shapes) ? gd.layout.shapes.filter((s) => !s._scrub) : [];
    // Rebuild threshold shapes without losing them: prefer last known from layout
    const scrubShape = {
      _scrub: true,
      type: "line",
      xref: "x",
      yref: "paper",
      x0: xPlot,
      x1: xPlot,
      y0: 0,
      y1: 1,
      line: { color: "rgba(10,132,255,.55)", width: 2 },
    };
    Plotly.relayout(gd, { shapes: [...baseShapes, scrubShape] });
  }

  function clearScrub() {
    if (!isHoveringChart) return;
    isHoveringChart = false;
    hoverReadoutValue = null;
    if (lastPayload) {
      updateReadout(lastPayload);
      if (readoutMeta) {
        readoutMeta.textContent =
          `Last measured reading ${lastPayload.latest_value.toFixed(2)} ft at ${formatTimestamp(lastPayload.latest_timestamp)}`;
      }
    }
    if (gd.layout && gd.layout.shapes) {
      const cleaned = gd.layout.shapes.filter((s) => !s._scrub);
      Plotly.relayout(gd, { shapes: cleaned });
    }
  }

  // Desktop mouse
  gd.addEventListener("mousemove", (e) => {
    if (IS_TOUCH_DEVICE) return;
    const sample = sampleAtClientX(e.clientX);
    if (sample) showScrub(sample);
  });
  gd.addEventListener("mouseleave", () => {
    if (!IS_TOUCH_DEVICE) clearScrub();
  });

  // Mobile: one finger on plot body = scrub (horizontal) without fighting page scroll when mostly vertical
  let scrubTouchId = null;
  let scrubStartX = 0;
  let scrubStartY = 0;
  let scrubLocked = null; // 'h' | 'v' | null

  gd.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 1) {
        scrubTouchId = null;
        scrubLocked = null;
        return;
      }
      // Let axis-pan handler own touches on the date axis
      const fullLayout = gd._fullLayout;
      if (fullLayout && fullLayout._size) {
        const rect = gd.getBoundingClientRect();
        const plotBottom = rect.top + fullLayout._size.t + fullLayout._size.h;
        if (e.touches[0].clientY >= plotBottom - 8) {
          scrubTouchId = null;
          return;
        }
      }
      scrubTouchId = e.touches[0].identifier;
      scrubStartX = e.touches[0].clientX;
      scrubStartY = e.touches[0].clientY;
      scrubLocked = null;
    },
    { passive: true, capture: true }
  );

  gd.addEventListener(
    "touchmove",
    (e) => {
      if (scrubTouchId == null || e.touches.length !== 1) return;
      const touch = e.touches[0];
      if (touch.identifier !== scrubTouchId) return;

      const dx = touch.clientX - scrubStartX;
      const dy = touch.clientY - scrubStartY;

      if (!scrubLocked) {
        if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return;
        // Lock to horizontal scrub vs vertical page scroll by dominant direction
        scrubLocked = Math.abs(dx) >= Math.abs(dy) ? "h" : "v";
      }

      if (scrubLocked === "v") {
        // Vertical intent → page scroll; clear scrub if any
        clearScrub();
        return;
      }

      // Horizontal scrub
      e.preventDefault();
      const sample = sampleAtClientX(touch.clientX);
      if (sample) showScrub(sample);
    },
    { passive: false, capture: true }
  );

  gd.addEventListener(
    "touchend",
    (e) => {
      if (scrubTouchId == null) return;
      const still = [...e.touches].some((t) => t.identifier === scrubTouchId);
      if (!still) {
        scrubTouchId = null;
        scrubLocked = null;
        clearScrub();
      }
    },
    { capture: true }
  );

  gd.addEventListener("touchcancel", () => {
    scrubTouchId = null;
    scrubLocked = null;
    clearScrub();
  }, { capture: true });
}

function formatHoverAxisTime(msOrVal) {
  try {
    const d = typeof msOrVal === "number" ? new Date(msOrVal) : new Date(msOrVal);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString("en-US", {
        timeZone: DISPLAY_TIME_ZONE,
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      });
    }
  } catch (_) {}
  return String(msOrVal);
}

function updateReadout(payload) {
  if (!payload) return;
  const { value } = interpolateNowValue(payload, Date.now());
  if (readout) {
    readout.innerHTML = `${value.toFixed(2)}<span class="unit">ft</span>`;
  }
  const { minValue, maxValue } = getEffectiveThresholds();
  const nearHigh = maxValue - value <= WARNING_MARGIN_FT;
  const nearLow = value - minValue <= WARNING_MARGIN_FT;
  if (readout) {
    readout.classList.toggle("near-limit-high", nearHigh && !nearLow);
    readout.classList.toggle("near-limit-low", nearLow);
  }
}

/**
 * Runs every second so the "now" marker (and the large readout) keep moving.
 */
function tickNowMarker() {
  if (!lastPayload || nowTraceIndex === null || !chartInitialized) return;
  const nowMs = Date.now();
  const { utcIso, value } = interpolateNowValue(lastPayload, nowMs);
  const displayX = toViewerPlotTimestamp(utcIso);
  Plotly.restyle("chart", { x: [[displayX]], y: [[value]] }, [nowTraceIndex]);
  if (!isHoveringChart) {
    updateReadout(lastPayload);
  }
}

/* ------------------------------------------------------------------ */
/*  Boot                                                               */
/* ------------------------------------------------------------------ */
initThresholdControls();
refresh();
setInterval(refresh, REFRESH_MS);
setInterval(tickNowMarker, 1000);
