"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
=======================================

In this step, we understand the data:

* Checking for missing values
* Looking at basic statistics
* Visualizing rainfall and temperature trends
* Comparing monsoon vs. non-monsoon periods
  """


import pandas as pd
import matplotlib.pyplot as plt

# ---- DATA LOAD  ----
df = pd.read_csv("../data/delhi_weather_raw.csv", parse_dates=["date"])
print("Shape (rows, columns):", df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe())

# ---- analysis MONTH/YEAR COLUMNS ----
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9])

print("\n--- Monsoon vs Non-Monsoon Average Rainfall ---")
print(df.groupby("is_monsoon")["rainfall"].mean())

# ---- VISUALIZATION 1: Temperature over time ----
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(df["date"], df["temp_mean"], color="orangered", linewidth=0.7)
axes[0].set_title("Delhi Daily Mean Temperature (2019-2024)")
axes[0].set_ylabel("Temperature (°C)")

# ---- VISUALIZATION 2: Rainfall over time ----
axes[1].bar(df["date"], df["rainfall"], color="steelblue", width=1)
axes[1].set_title("Delhi Daily Rainfall (2019-2024)")
axes[1].set_ylabel("Rainfall (mm)")

# ---- VISUALIZATION 3: Monthly average rainfall (seasonal pattern) ----
monthly_rain = df.groupby("month")["rainfall"].mean()
axes[2].bar(monthly_rain.index, monthly_rain.values, color="seagreen")
axes[2].set_title("Average Rainfall by Month (Monsoon pattern dikhna chahiye June-Sept)")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Avg Rainfall (mm)")
axes[2].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig("../outputs/eda_plots.png", dpi=120)
print("\nGraphs save ho gaye: ../outputs/eda_plots.png")

# ---- CORRELATION HEATMAP (which feature related to which) ----
import numpy as np
numeric_cols = ["temp_max", "temp_min", "temp_mean", "rainfall", "wind_speed",
                 "humidity", "pressure", "cloud_cover"]
corr = df[numeric_cols].corr()

fig2, ax2 = plt.subplots(figsize=(8, 6))
im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax2.set_xticks(range(len(numeric_cols)))
ax2.set_yticks(range(len(numeric_cols)))
ax2.set_xticklabels(numeric_cols, rotation=45, ha="right")
ax2.set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax2.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../outputs/correlation_heatmap.png", dpi=120)
print("Correlation heatmap save ho gaya: ../outputs/correlation_heatmap.png")
