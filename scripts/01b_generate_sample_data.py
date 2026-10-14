"""
STEP 1B: SAMPLE DATA GENERATOR (For Demo/Testing Only)
======================================================

IMPORTANT: This is NOT real data. It is only used so that you can
see and test the complete EDA and ML pipeline flow if you currently
do not have internet access or have not run 01_data_collection.py
on your laptop.

When you run 01_data_collection.py on your laptop, it will generate
the REAL data file 'delhi_weather_raw.csv', and all the remaining
scripts (02, 03, 04) will automatically work with that file.

It follows a realistic Delhi monsoon pattern:

* Higher rainfall & humidity from June to September
* Highest temperatures in May
* Lowest temperatures in December and January
  """
  

from common import DATA_DIR, prepare_dirs
import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
n = len(dates)
day_of_year = dates.dayofyear
month = dates.month

# Temperature: seasonal sine wave + noise (Delhi jaisa pattern)
temp_mean = 25 + 12 * np.sin((day_of_year - 80) * 2 * np.pi / 365) + np.random.normal(0, 2, n)
temp_max = temp_mean + np.random.uniform(4, 8, n)
temp_min = temp_mean - np.random.uniform(4, 8, n)

# Monsoon months (June-Sept = month 6,7,8,9) mein zyada rainfall
is_monsoon = np.isin(month, [6, 7, 8, 9])
rainfall = np.where(
    is_monsoon,
    np.random.exponential(scale=8, size=n) * np.random.binomial(1, 0.5, n),
    np.random.exponential(scale=0.5, size=n) * np.random.binomial(1, 0.1, n),
)

humidity = np.where(
    is_monsoon,
    np.random.uniform(60, 90, n),
    np.random.uniform(25, 55, n),
)

pressure = 1005 + 5 * np.cos((day_of_year) * 2 * np.pi / 365) + np.random.normal(0, 1.5, n)
wind_speed = np.random.uniform(5, 20, n) + is_monsoon * np.random.uniform(0, 5, n)
cloud_cover = np.where(is_monsoon, np.random.uniform(40, 90, n), np.random.uniform(0, 40, n))

df = pd.DataFrame({
    "date": dates,
    "temp_max": temp_max.round(1),
    "temp_min": temp_min.round(1),
    "temp_mean": temp_mean.round(1),
    "rainfall": rainfall.round(1),
    "rain": rainfall.round(1),
    "wind_speed": wind_speed.round(1),
    "humidity": humidity.round(1),
    "pressure": pressure.round(1),
    "cloud_cover": cloud_cover.round(1),
})

prepare_dirs(DATA_DIR)
output = DATA_DIR / "delhi_weather_raw.csv"
df.to_csv(output, index=False)
print(f"SYNTHETIC sample data ready: {len(df)} rows -> {output} (this overwrites the raw CSV)")
print(df.head())
