"""
## STEP 1: DATA COLLECTION

========================

This script downloads historical daily weather data for Delhi using the **Open-Meteo API**, which is completely free and does not require an API key.

**API Documentation:** https://open-meteo.com/en/docs/historical-weather-api

"""

import logging
import requests
import pandas as pd
from common import DATA_DIR, prepare_dirs

# ---- CONFIGURATION ----

# Coordinates for Delhi. If you need data for another city,

# simply change the latitude and longitude.

LATITUDE = 28.6139
LONGITUDE = 77.2090
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---- API CALL ----
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",       # rainfall
        "rain_sum",
        "wind_speed_10m_max",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "cloud_cover_mean",
    ]),
    "timezone": "Asia/Kolkata",
}

logging.info("Downloading historical weather from Open-Meteo")
try:
    response = requests.get(url, params=params, timeout=(10, 90))
    response.raise_for_status()
    data = response.json()
except (requests.RequestException, ValueError) as exc:
    raise SystemExit(f"Weather download failed; existing data was left unchanged: {exc}") from exc
if "daily" not in data or not data["daily"].get("time"):
    raise SystemExit("Open-Meteo returned no daily observations; existing data was left unchanged.")

# ---- JSON convert to DataFrame ----
df = pd.DataFrame(data["daily"])
df.rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "rainfall",
    "rain_sum": "rain",
    "wind_speed_10m_max": "wind_speed",
    "relative_humidity_2m_mean": "humidity",
    "surface_pressure_mean": "pressure",
    "cloud_cover_mean": "cloud_cover",
}, inplace=True)

required_columns = {
    "date", "temp_max", "temp_min", "temp_mean", "rainfall", "rain", "wind_speed",
    "humidity", "pressure", "cloud_cover",
}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise SystemExit(
        "Open-Meteo response omitted expected fields: " + ", ".join(sorted(missing_columns))
        + ". Existing weather data was left unchanged."
    )

df["date"] = pd.to_datetime(df["date"])

# ---- SAVE  ----
prepare_dirs(DATA_DIR)
output_path = DATA_DIR / "delhi_weather_raw.csv"
df.to_csv(output_path, index=False)

print(f"Done! {len(df)} din ka data save ho gaya: {output_path}")
print(df.head())
