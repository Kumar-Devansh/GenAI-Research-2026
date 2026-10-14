"""
STEP 3: FEATURE ENGINEERING
===========================

We convert the raw data into "features" that can be used by the ML model.

Idea: To predict tomorrow's weather, data from the previous few days
(lag features) and date-based information can be useful.

We will create the target: Predicting tomorrow's rainfall (1-day-ahead forecast).
"""


from common import DATA_DIR, load_weather
import pandas as pd
import numpy as np

df = load_weather(DATA_DIR / "delhi_weather_raw.csv")

# ---- 1. DATE-BASED FEATURES ----
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

# Cyclical encoding: Converting the month into sin/cos is better

# because December (12) and January (1) are actually "close" to each other,

# but numerically, 12 and 1 appear far apart. Sin/cos encoding fixes this problem.

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ---- 2. LAG FEATURES (Data from Previous Days) ----

# "What was the temperature/rainfall yesterday?" -> This helps predict today's weather.

for lag in [1, 2, 3, 7]:
    df[f"temp_mean_lag{lag}"] = df["temp_mean"].shift(lag)
    df[f"rainfall_lag{lag}"] = df["rainfall"].shift(lag)
    df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)
    df[f"pressure_lag{lag}"] = df["pressure"].shift(lag)

# ---- 3. ROLLING AVERAGE FEATURES ( Average of the previous 7 days) ----
df["temp_mean_roll7"] = df["temp_mean"].rolling(window=7).mean()
df["rainfall_roll7"] = df["rainfall"].rolling(window=7).mean()
df["humidity_roll7"] = df["humidity"].rolling(window=7).mean()

# ---- 4. TARGET VARIABLE ----

# We will predict the next day's (tomorrow's) mean temperature and rainfall.

df["target_temp_next_day"] = df["temp_mean"].shift(-1)
df["target_rainfall_next_day"] = df["rainfall"].shift(-1)

# ---- 5. REMOVE MISSING VALUES ----

# The first few rows will have NaN values in the lag features (because previous data is not available)

# and the last row will have a NaN target (because the next day's data is not available).

before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)
print(f"Rows before cleaning: {before}, after: {after} (removed {before-after} rows with missing values)")

# ---- SAVE FEATURES ----
df.to_csv(DATA_DIR / "delhi_weather_features.csv", index=False)
print(f"Feature-engineered data saved: {DATA_DIR / 'delhi_weather_features.csv'}")
print(f"Total features: {df.shape[1]} columns, {df.shape[0]} rows")
print("\nColumns:", list(df.columns))
