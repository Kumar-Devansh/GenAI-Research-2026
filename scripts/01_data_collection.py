"""
## STEP 1: DATA COLLECTION

========================

This script downloads historical daily weather data for Delhi using the **Open-Meteo API**, which is completely free and does not require an API key.

**API Documentation:** https://open-meteo.com/en/docs/historical-weather-api

"""

import requests
import pandas as pd
from datetime import date

# ---- CONFIGURATION ----

# Coordinates for Delhi. If you need data for another city,

# simply change the latitude and longitude.

LATITUDE = 28.6139
LONGITUDE = 77.2090
START_DATE = "2019-01-01"   # 5 saal ka data lenge
END_DATE = "2024-12-31"

# ---- API CALL ----
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",       # rainfall
        "rain_sum",
        "windspeed_10m_max",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "cloudcover_mean",
    ],
    "timezone": "Asia/Kolkata",
}

print("Data download ho raha hai Open-Meteo se... thoda wait karo.")
response = requests.get(url, params=params)
response.raise_for_status()   # if error occures stop the script and show error
data = response.json()

# ---- JSON convert to DataFrame ----
df = pd.DataFrame(data["daily"])
df.rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "rainfall",
    "rain_sum": "rain",
    "windspeed_10m_max": "wind_speed",
    "relative_humidity_2m_mean": "humidity",
    "surface_pressure_mean": "pressure",
    "cloudcover_mean": "cloud_cover",
}, inplace=True)

df["date"] = pd.to_datetime(df["date"])

# ---- SAVE  ----
output_path = "../data/delhi_weather_raw.csv"
df.to_csv(output_path, index=False)

print(f"Done! {len(df)} din ka data save ho gaya: {output_path}")
print(df.head())
