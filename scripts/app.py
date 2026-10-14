"""Streamlit front end for the Delhi Weather Intelligence research platform."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# The app is also launchable as ``streamlit run scripts/app.py`` from the repo root.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import predict_next_day, search_knowledge  # noqa: E402
from weather_sources import WeatherSourceError, imd_current_weather, open_meteo_forecast  # noqa: E402


def file_version(path: Path) -> int | None:
    """Cache key that changes when a local data or result file is regenerated."""
    return path.stat().st_mtime_ns if path.is_file() else None


st.set_page_config(
    page_title="Atmos | Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS = {
    "Overview": "◉",
    "Weather intelligence": "⌁",
    "Monsoon analytics": "≋",
    "Satellite intelligence": "▧",
    "AI assistant": "✳",
    "Model laboratory": "⌗",
    "Knowledge base": "⌕",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --ink: #eaf2fa; --muted: #91a7bb; --dim: #6d8297;
      --navy: #071321; --panel: #0c1b2b; --panel-2: #102237;
      --line: rgba(152, 184, 211, .13); --cyan: #67d9e8;
      --blue: #77a8ff; --mint: #76d6bb;
    }
    html, body, [class*="css"] { font-family: 'Manrope', 'Segoe UI', sans-serif; }
    .stApp {
      color: var(--ink);
      background:
        radial-gradient(ellipse at 72% -8%, rgba(49, 105, 157, .18), transparent 42%),
        radial-gradient(ellipse at 8% 35%, rgba(16, 87, 115, .08), transparent 32%),
        #071321;
    }
    [data-testid="stHeader"] { background: rgba(7, 19, 33, .72); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #091726 0%, #081522 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    .block-container { max-width: 1500px; padding: 2rem 3.2rem 3.5rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
    h1 { font-weight: 700; }
    p, li, label { color: #c7d4e0; }
    [data-testid="stCaptionContainer"] p, .muted { color: var(--muted) !important; }
    [data-testid="stMetric"] {
      background: linear-gradient(150deg, rgba(16, 34, 53, .93), rgba(11, 27, 43, .95));
      border: 1px solid var(--line); border-radius: 13px; padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .79rem; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace; }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
      border-color: var(--line) !important; border-radius: 14px !important;
      background: rgba(12, 27, 43, .76);
    }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
      border: 1px solid rgba(103, 217, 232, .34); color: #d8fbff;
      background: linear-gradient(120deg, rgba(34, 102, 129, .48), rgba(31, 66, 108, .6));
      border-radius: 9px; font-weight: 700; transition: .18s ease;
    }
    [data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
      border-color: var(--cyan); background: rgba(40, 108, 134, .7); color: white;
      transform: translateY(-1px);
    }
    [data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
      background-color: #0d2033 !important; border-color: var(--line) !important;
      color: var(--ink) !important; border-radius: 9px !important;
    }
    [data-testid="stRadio"] label { color: #a9bbcc !important; }
    [data-testid="stTabs"] button { color: var(--muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--cyan); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stAlert"] { border-radius: 10px; }
    .brand-lockup { display:flex; align-items:center; gap:.75rem; padding:.35rem .1rem 1.5rem; }
    .brand-mark {
      width:38px;height:38px;display:grid;place-items:center;border-radius:11px;
      color:#8be5ed;font-size:22px;background:linear-gradient(145deg,#163d58,#0c263a);
      border:1px solid rgba(103,217,232,.25); box-shadow:0 8px 26px rgba(17,109,135,.14);
    }
    .brand-name { color:#eff8ff;font-size:1rem;font-weight:800;letter-spacing:-.04em; }
    .brand-sub { color:#7890a4;font-size:.61rem;letter-spacing:.16em;text-transform:uppercase;margin-top:2px; }
    .side-label { color:#647d93;font:500 .64rem 'DM Mono',monospace;letter-spacing:.16em;
      text-transform:uppercase;margin:1.1rem 0 .45rem; }
    .page-kicker { color:var(--cyan);font:500 .69rem 'DM Mono',monospace;letter-spacing:.15em;
      text-transform:uppercase;margin:0 0 .5rem; }
    .page-title { color:#edf6ff;font-size:2rem;font-weight:700;letter-spacing:-.055em;margin:0; }
    .page-subtitle { color:#8ea4b8;font-size:.92rem;margin:.48rem 0 1.35rem; }
    .hero {
      position:relative; overflow:hidden; min-height:265px; padding:2rem 2.15rem;
      border:1px solid rgba(119,168,255,.2); border-radius:17px;
      background:
        radial-gradient(ellipse at 82% 48%, rgba(67,153,196,.24), transparent 33%),
        radial-gradient(ellipse at 67% 15%, rgba(58,103,163,.27), transparent 38%),
        linear-gradient(115deg,#112a43 0%,#0c2035 55%,#10253b 100%);
      box-shadow:0 18px 60px rgba(0,0,0,.14);
    }
    .hero:after { content:"";position:absolute;right:-7%;top:-50%;width:58%;height:190%;
      border:1px solid rgba(132,209,233,.10);border-radius:50%;transform:rotate(-23deg);
      box-shadow:0 0 0 36px rgba(132,209,233,.025),0 0 0 74px rgba(132,209,233,.02);pointer-events:none; }
    .hero-label { color:#a9c6db;font:500 .69rem 'DM Mono',monospace;letter-spacing:.13em;text-transform:uppercase; }
    .hero-temp { font-size:4rem;line-height:1.05;letter-spacing:-.08em;font-weight:700;color:#f1f8ff;margin:.55rem 0 .4rem; }
    .hero-copy { color:#a6bbcd;font-size:.85rem; }
    .hero-chip { display:inline-flex;gap:.45rem;align-items:center;margin-top:1.25rem;padding:.45rem .7rem;
      border:1px solid rgba(135,190,219,.22);border-radius:99px;color:#c4d8e8;
      background:rgba(4,17,30,.25);font:400 .69rem 'DM Mono',monospace; }
    .live-dot { width:7px;height:7px;border-radius:50%;background:#7ad7be;display:inline-block; }
    .section-heading { color:#e7f1f9;font-size:1.08rem;font-weight:700;letter-spacing:-.025em;margin:.55rem 0 .2rem; }
    .section-note { color:#8298ac;font-size:.78rem;margin:0 0 1rem; }
    .eyebrow { color:#7790a5;font:500 .64rem 'DM Mono',monospace;letter-spacing:.13em;text-transform:uppercase; }
    .small-value { color:#eaf4fc;font-size:1.35rem;font-weight:700;letter-spacing:-.04em;margin-top:.35rem; }
    .status-line { padding:.8rem .95rem;border-radius:10px;background:rgba(104,216,232,.055);
      border:1px solid rgba(104,216,232,.13);color:#a9c3d4;font-size:.8rem; }
    .footer-note { color:#60768b;font:400 .65rem 'DM Mono',monospace;margin-top:2rem; }
    @media (max-width: 900px) { .block-container {padding:1.35rem 1.2rem 2.5rem;} .hero {padding:1.5rem;min-height:235px;} }
    @media (max-width: 560px) { .page-title {font-size:1.65rem;} .hero-temp {font-size:3.3rem;} }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after {scroll-behavior:auto !important;transition:none !important;animation:none !important;} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_weather(modified_at: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / "delhi_weather_raw.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" not in frame or frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_results(filename: str, modified_at: int | None = None) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def heading(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{kicker}</div><h1 class="page-title">{title}</h1>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str) -> None:
    st.markdown(
        f'<div class="section-heading">{title}</div><div class="section-note">{note}</div>',
        unsafe_allow_html=True,
    )


def chart_period(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    choice = st.selectbox("Time window", ["Past 90 days", "Past year", "Full archive"], key=key)
    if frame.empty:
        return frame
    days = {"Past 90 days": 90, "Past year": 365}.get(choice)
    if days is None:
        return frame
    return frame[frame["date"] >= frame["date"].max() - pd.Timedelta(days=days)]


def render_chart(frame: pd.DataFrame, column: str, title: str, unit: str, key: str) -> None:
    if column not in frame:
        st.info(f"{title} is not present in the weather archive.")
        return
    data = chart_period(frame[["date", column]].dropna(), key).set_index("date")
    if data.empty:
        st.info(f"No {title.lower()} values are available for this period.")
        return
    st.caption(f"{title} · {unit}")
    st.line_chart(data, height=270, color="#69cddd")


def render_overview(frame: pd.DataFrame) -> None:
    heading("Weather intelligence / 01", "Delhi climate desk", "A research view of atmospheric conditions, model outlooks, and monsoon signals.")
    if frame.empty:
        st.error("Weather archive is unavailable. Run the Part 01 data collection step to populate this dashboard.")
        return

    latest = frame.iloc[-1]
    stamp = latest["date"].strftime("%d %b %Y")
    start = frame["date"].min().strftime("%Y")
    end = frame["date"].max().strftime("%Y")
    location_col, source_col = st.columns([1, 2])
    with location_col:
        st.selectbox("Study location", ["Delhi, India"], help="Only Delhi is available in the current dataset.")
    with source_col:
        st.markdown(f'<div class="footer-note" style="text-align:right;margin-top:2.2rem">ARCHIVE {start}–{end} &nbsp;·&nbsp; LATEST OBSERVATION {stamp} &nbsp;·&nbsp; OPEN-METEO</div>', unsafe_allow_html=True)

    temp = latest.get("temp_mean")
    temp_label = f"{temp:.1f}°" if pd.notna(temp) else "—"
    st.markdown(
        f'<div class="hero"><div class="hero-label">Delhi · latest daily observation</div>'
        f'<div class="hero-temp">{temp_label}<span style="font-size:1.2rem;letter-spacing:0;color:#9fb9cd"> C</span></div>'
        f'<div class="hero-copy">Mean air temperature in the historical archive · {stamp}</div>'
        f'<div class="hero-chip"><span class="live-dot"></span> HISTORICAL ARCHIVE · NOT LIVE</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    metrics = [
        ("Rainfall", "rainfall", "mm", "Precipitation total"),
        ("Relative humidity", "humidity", "%", "Daily mean"),
        ("Wind speed", "wind_speed", "km/h", "Daily maximum"),
        ("Cloud cover", "cloud_cover", "%", "Daily mean"),
    ]
    cols = st.columns(4)
    for col, (label, key, unit, detail) in zip(cols, metrics):
        value = latest.get(key)
        with col:
            st.metric(label, f"{value:.1f} {unit}" if pd.notna(value) else "Unavailable")
            st.caption(detail)

    left, right = st.columns([1.45, 1])
    with left:
        section_title("Temperature signal", "Daily mean temperature · archive trend")
        render_chart(frame, "temp_mean", "Daily mean temperature", "°C", "overview_temp_window")
    with right:
        section_title("Rainfall pattern", "Recent precipitation observations")
        render_chart(frame, "rainfall", "Daily rainfall", "mm", "overview_rain_window")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_forecast, col_context = st.columns([1, 1.45])
    with col_forecast:
        with st.container(border=True):
            section_title("Research-model estimate", "Next day after the local historical archive · not a live forecast")
            st.markdown('<div class="status-line">This saved LSTM/GRU uses the archive’s final 14 days. Use the separate weather forecast below for current provider output.</div>', unsafe_allow_html=True)
            archive_version = file_version(DATA_DIR / "delhi_weather_raw.csv")
            if st.button("Generate next-day outlook", key="overview_forecast"):
                st.session_state.pop("weather_forecast", None)
                with st.spinner("Running the saved forecasting model…"):
                    try:
                        st.session_state["weather_forecast"] = predict_next_day()
                        st.session_state["weather_forecast_version"] = archive_version
                        st.session_state.pop("weather_forecast_error", None)
                    except Exception as exc:
                        st.session_state["weather_forecast_error"] = str(exc)
                        st.session_state["weather_forecast_error_version"] = archive_version
            if st.session_state.get("weather_forecast") and st.session_state.get("weather_forecast_version") == archive_version:
                result = st.session_state["weather_forecast"]
                st.metric(f"Mean temperature · {result['forecast_date']}", f"{result['predicted_mean_temp_c']:.1f} °C")
                st.caption(f"Saved model: {result['model']} · archive through {result.get('latest_input_date', 'unknown')}")
            elif not st.session_state.get("weather_forecast_error") or st.session_state.get("weather_forecast_version") != archive_version:
                st.caption("Generate a forecast to see the saved model’s output for the current archive.")
            if st.session_state.get("weather_forecast_error") and st.session_state.get("weather_forecast_error_version") == archive_version:
                st.warning(f"Forecast unavailable: {st.session_state['weather_forecast_error']}")
            st.divider()
            st.markdown("**Provider weather · Delhi**")
            if st.button("Fetch current model conditions and 7-day forecast", key="live_weather"):
                try:
                    st.session_state["provider_weather"] = open_meteo_forecast()
                    st.session_state.pop("provider_weather_error", None)
                except WeatherSourceError as exc:
                    st.session_state["provider_weather_error"] = str(exc)
            live = st.session_state.get("provider_weather")
            if live:
                current = live["current"]
                a, b, c = st.columns(3)
                a.metric("Model temperature", f"{current.get('temperature_2m', '—')} °C")
                b.metric("Precipitation", f"{current.get('precipitation', '—')} mm")
                c.metric("Relative humidity", f"{current.get('relative_humidity_2m', '—')}%")
                st.caption(f"{live['provider']} {live['product']} · as of {current.get('time', 'time unavailable')} · fetched {live['fetched_at_utc']} · source: {live['source_url']}")
                daily = live.get("daily", {})
                if daily.get("time"):
                    st.dataframe(pd.DataFrame({"Date": daily["time"], "High °C": daily.get("temperature_2m_max", []),
                                               "Low °C": daily.get("temperature_2m_min", []),
                                               "Precip. mm": daily.get("precipitation_sum", []),
                                               "Rain probability %": daily.get("precipitation_probability_max", [])}),
                                 hide_index=True, use_container_width=True)
            if st.session_state.get("provider_weather_error"):
                st.warning(f"Provider forecast unavailable: {st.session_state['provider_weather_error']}")
            with st.expander("Official IMD station observations (optional)"):
                st.caption("Requires IMD_API_AUTH_HEADER, IMD_API_AUTH_VALUE, and IMD_STATION_ID configured per your IMD account. The public reference does not specify an authentication scheme. IMD station observations are separate from the Open-Meteo model output above.")
                if st.button("Fetch IMD station data", key="imd_weather"):
                    try:
                        st.session_state["imd_weather"] = imd_current_weather()
                        st.session_state.pop("imd_weather_error", None)
                    except WeatherSourceError as exc:
                        st.session_state["imd_weather_error"] = str(exc)
                if st.session_state.get("imd_weather"):
                    st.json(st.session_state["imd_weather"])
                if st.session_state.get("imd_weather_error"):
                    st.info(st.session_state["imd_weather_error"])
    with col_context:
        with st.container(border=True):
            section_title("Monsoon context", "Observed archive summary · June through September")
            monthly = frame.assign(month=frame["date"].dt.month)
            monsoon = monthly[monthly["month"].isin([6, 7, 8, 9])]
            outside = monthly[~monthly["month"].isin([6, 7, 8, 9])]
            avg_monsoon = monsoon["rainfall"].mean() if "rainfall" in monsoon else float("nan")
            avg_other = outside["rainfall"].mean() if "rainfall" in outside else float("nan")
            c1, c2 = st.columns(2)
            c1.metric("Monsoon daily mean", f"{avg_monsoon:.1f} mm" if pd.notna(avg_monsoon) else "Unavailable")
            c2.metric("Other months daily mean", f"{avg_other:.1f} mm" if pd.notna(avg_other) else "Unavailable")
            st.caption("Calculated from available historical rows; descriptive only, not a forecast.")


def render_weather(frame: pd.DataFrame) -> None:
    heading("Observations / 02", "Weather intelligence", "Explore the daily atmospheric record across temperature, moisture, wind, and cloud cover.")
    if frame.empty:
        st.error("Weather archive is unavailable.")
        return
    columns = [
        ("Mean temperature", "temp_mean", "°C"),
        ("Maximum temperature", "temp_max", "°C"),
        ("Minimum temperature", "temp_min", "°C"),
        ("Rainfall", "rainfall", "mm"),
        ("Humidity", "humidity", "%"),
        ("Wind speed", "wind_speed", "km/h"),
        ("Surface pressure", "pressure", "hPa"),
        ("Cloud cover", "cloud_cover", "%"),
    ]
    labels = [label for label, col, _ in columns if col in frame]
    if not labels:
        st.info("No supported weather measurements are present in this archive.")
        return
    selected = st.selectbox("Atmospheric variable", labels)
    chosen = next(item for item in columns if item[0] == selected)
    left, right = st.columns([1.5, 1])
    with left:
        render_chart(frame, chosen[1], chosen[0], chosen[2], "weather_window")
    with right:
        section_title("Latest reading", f"Daily value · {frame['date'].max():%d %b %Y}")
        value = frame.iloc[-1].get(chosen[1])
        st.markdown(f'<div class="small-value">{value:.2f} <span style="font-size:.8rem;color:#91a7bb">{chosen[2]}</span></div>' if pd.notna(value) else '<div class="small-value">Unavailable</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        numeric = pd.to_numeric(frame[chosen[1]], errors="coerce").dropna()
        if not numeric.empty:
            st.metric("Archive median", f"{numeric.median():.2f} {chosen[2]}")
            st.metric("Archive range", f"{numeric.min():.2f} – {numeric.max():.2f} {chosen[2]}")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    section_title("Observation table", "Most recent records in the local historical dataset")
    display_cols = [c for c in ["date", "temp_min", "temp_mean", "temp_max", "rainfall", "humidity", "wind_speed", "pressure", "cloud_cover"] if c in frame]
    st.dataframe(frame[display_cols].tail(14).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.caption("Source: Open-Meteo Historical Weather API · daily archive. Values describe archived observations, not live conditions.")


def render_monsoon(frame: pd.DataFrame) -> None:
    heading("Seasonal signals / 03", "Monsoon analytics", "Historical rainfall patterns for Delhi, with clear separation between observed data and interpretation.")
    if frame.empty or "rainfall" not in frame:
        st.error("Rainfall observations are not available in the local archive.")
        return
    data = frame.assign(year=frame["date"].dt.year, month=frame["date"].dt.month)
    monsoon = data[data["month"].between(6, 9)]
    annual = monsoon.groupby("year", as_index=False)["rainfall"].sum().rename(columns={"rainfall": "Monsoon rainfall (mm)"})
    left, right = st.columns([1.35, 1])
    with left:
        section_title("Seasonal rainfall", "June–September total by year · mm")
        if annual.empty:
            st.info("No monsoon season rows are available.")
        else:
            st.bar_chart(annual.set_index("year"), height=300, color="#62c8d5")
    with right:
        section_title("Monthly climatology", "Average daily rainfall by calendar month · mm/day")
        monthly = data.groupby("month")["rainfall"].mean().reindex(range(1, 13))
        monthly.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        st.bar_chart(monthly, height=300, color="#6e9dec")
    c1, c2, c3 = st.columns(3)
    c1.metric("Years represented", str(int(monsoon["year"].nunique())))
    c2.metric("Mean monsoon total", f"{annual.iloc[:, 1].mean():.1f} mm" if not annual.empty else "Unavailable")
    c3.metric("Wettest archived monsoon", f"{annual.iloc[:, 1].max():.1f} mm" if not annual.empty else "Unavailable")
    st.caption("Totals are grouped by calendar year and June–September. Incomplete edge years are not adjusted. No official alert feed is connected.")


def render_satellite() -> None:
    heading("Remote sensing / 04", "Satellite intelligence", "A workspace prepared for imagery and segmentation products; current samples are procedural demonstrations.")
    st.markdown('<div class="status-line">No live satellite imagery, georeferenced tiles, or segmentation service is connected in this Part 01 app.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.info("The images below are synthetically generated sky-pattern samples used by the CNN proof of concept. They are not satellite scenes or current weather evidence.")
    categories = [("clear_sky", "Clear-sky sample"), ("partly_cloudy", "Partly cloudy sample"), ("heavy_clouds", "Dense-cloud sample")]
    cols = st.columns(3)
    available = False
    for col, (category, label) in zip(cols, categories):
        folder = DATA_DIR / "sky_images" / category
        images = sorted(folder.glob("*.png")) if folder.is_dir() else []
        with col:
            with st.container(border=True):
                section_title(label, "Synthetic image · 64 × 64")
                if images:
                    available = True
                    st.image(str(images[len(images) // 2]), use_column_width=True)
                else:
                    st.caption("No local sample image found.")
    if not available:
        st.warning("Generate the synthetic sample set with scripts/09a_generate_sky_images.py to view the CNN demo images.")
    cnn_metrics_path = OUTPUT_DIR / "cnn_classification_comparison.csv"
    metrics = load_results("cnn_classification_comparison.csv", file_version(cnn_metrics_path))
    if not metrics.empty and "Accuracy" in metrics:
        st.metric("Saved CNN test accuracy", f"{metrics.iloc[0]['Accuracy'] * 100:.1f}%")
        st.caption("This is a held-out score on the synthetic dataset only; it does not measure satellite-image performance.")
    else:
        st.caption("No CNN evaluation result is available yet. Train with scripts/09b_cnn_classification.py to generate one.")


def render_assistant() -> None:
    heading("Research copilot / 05", "AI assistant", "Ask about a forecast or monsoon knowledge. Responses depend on the configured local Ollama model and tools.")
    st.markdown('<div class="status-line">Forecasts and knowledge retrieval are tool-backed. The assistant may be unavailable if Ollama or its model is not running.</div>', unsafe_allow_html=True)
    st.caption("The existing agent handles each prompt independently; previous turns remain visible here as a session log.")
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []
    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about Delhi weather or the Southwest Monsoon…")
    if prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Consulting the local assistant…"):
                try:
                    from agent import run_agent
                    answer = run_agent(prompt)
                except Exception as exc:
                    answer = f"The assistant could not respond: {exc}"
            st.markdown(answer)
        st.session_state.assistant_messages.append({"role": "assistant", "content": answer})
    if st.session_state.assistant_messages and st.button("Clear conversation", key="clear_assistant"):
        st.session_state.assistant_messages = []
        st.rerun()


def render_models() -> None:
    heading("Evaluation / 06", "Model laboratory", "Review saved model evaluations across forecasting families and image classification.")
    st.markdown('<div class="status-line">Metrics below are read from saved output CSVs. These files do not currently record their evaluation date range or run provenance.</div>', unsafe_allow_html=True)
    families = [
        ("Feature-based ML", "model_comparison.csv", "Model", "R2", "RMSE"),
        ("Statistical time series", "timeseries_model_comparison.csv", "Model", "R2", "RMSE"),
        ("Sequence networks", "deep_learning_comparison.csv", "Model", "R2", "RMSE"),
        ("Multivariate sequence", "multivariate_dl_comparison.csv", "model", "R2", "RMSE"),
    ]
    tabs = st.tabs([family[0] for family in families])
    for tab, (label, filename, model_col, r2_col, rmse_col) in zip(tabs, families):
        result = load_results(filename, file_version(OUTPUT_DIR / filename))
        with tab:
            if result.empty:
                st.info(f"No saved {label.lower()} results. Run the associated training script first.")
                continue
            if r2_col not in result:
                r2_col = next((col for col in result if col.lower() == "r2"), r2_col)
            if model_col not in result or r2_col not in result:
                st.warning(f"The saved {filename} does not contain the expected model and R² columns.")
                st.dataframe(result, use_container_width=True, hide_index=True)
                continue
            left, right = st.columns([1.1, .9])
            with left:
                section_title(f"{label} · R²", "Higher is better; negative values are retained.")
                chart = result.set_index(model_col)[[r2_col]].rename(columns={r2_col: "R²"})
                st.bar_chart(chart, height=300, color="#6fcbd8")
            with right:
                section_title("Evaluation metrics", "Values as saved by the training scripts")
                st.dataframe(result, use_container_width=True, hide_index=True)
    cnn = load_results("cnn_classification_comparison.csv", file_version(OUTPUT_DIR / "cnn_classification_comparison.csv"))
    if not cnn.empty:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        section_title("Computer vision · synthetic image classification", "A separate task; accuracy is not comparable with forecast R².")
        st.dataframe(cnn, use_container_width=True, hide_index=True)


def render_knowledge() -> None:
    heading("Source retrieval / 07", "Knowledge base", "Search the local indexed monsoon documents and inspect the source passages returned by retrieval.")
    st.markdown('<div class="status-line">Results come from the local Chroma collection. This repository currently includes a small starter note, not a comprehensive official document corpus.</div>', unsafe_allow_html=True)
    with st.form("knowledge_search", clear_on_submit=False):
        query = st.text_input("Search query", placeholder="e.g. monsoon withdrawal or heavy rainfall thresholds")
        submitted = st.form_submit_button("Search knowledge")
    if submitted:
        if not query.strip():
            st.warning("Enter a search phrase to continue.")
        else:
            with st.spinner("Searching indexed passages…"):
                try:
                    result = search_knowledge(query, k=5)
                    st.session_state["knowledge_result"] = result
                    st.session_state.pop("knowledge_error", None)
                except Exception as exc:
                    st.session_state["knowledge_error"] = str(exc)
    if st.session_state.get("knowledge_error"):
        st.warning(f"Knowledge search unavailable: {st.session_state['knowledge_error']}")
    result = st.session_state.get("knowledge_result", "")
    if result:
        section_title("Retrieved passages", "Source labels are taken from the local knowledge base metadata")
        passages = [part.strip() for part in result.split("\n---\n") if part.strip()]
        for index, passage in enumerate(passages, start=1):
            with st.container(border=True):
                st.markdown(f'<div class="eyebrow">PASSAGE {index:02d}</div>', unsafe_allow_html=True)
                st.write(passage)
    elif not st.session_state.get("knowledge_error"):
        st.caption("Search results will appear here with their indexed source labels.")


with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">◌</div><div>'
        '<div class="brand-name">ATMOS</div><div class="brand-sub">Weather intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">Research workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        list(NAV_ITEMS),
        format_func=lambda item: f"{NAV_ITEMS[item]}   {item}",
        label_visibility="collapsed",
        key="active_page",
    )
    st.markdown('<div class="side-label" style="margin-top:1.7rem">Study configuration</div>', unsafe_allow_html=True)
    st.selectbox("Region", ["Delhi · 28.61°N, 77.21°E"], label_visibility="visible")
    st.markdown('<div class="status-line" style="margin-top:.35rem"><span class="live-dot"></span> &nbsp;Historical archive connected</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-note">PART 01 &nbsp;·&nbsp; RESEARCH BUILD</div>', unsafe_allow_html=True)


weather_path = DATA_DIR / "delhi_weather_raw.csv"
weather = load_weather(file_version(weather_path))
if page == "Overview":
    render_overview(weather)
elif page == "Weather intelligence":
    render_weather(weather)
elif page == "Monsoon analytics":
    render_monsoon(weather)
elif page == "Satellite intelligence":
    render_satellite()
elif page == "AI assistant":
    render_assistant()
elif page == "Model laboratory":
    render_models()
elif page == "Knowledge base":
    render_knowledge()

st.markdown('<div class="footer-note">ATMOS CLIMATE INTELLIGENCE &nbsp;·&nbsp; Delhi research archive &nbsp;·&nbsp; Part 01</div>', unsafe_allow_html=True)
