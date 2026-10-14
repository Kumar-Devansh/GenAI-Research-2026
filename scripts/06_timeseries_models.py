
"""
STEP 5: TIME-SERIES FORECASTING MODELS
======================================

Now we will try traditional statistical time-series models:

1. ARIMA   - AutoRegressive Integrated Moving Average
2. SARIMA  - Seasonal ARIMA (captures seasonal patterns such as monsoon effects)
3. Prophet - Developed by Facebook/Meta, popular for modeling trends and seasonality

DIFFERENCE FROM ML MODELS: These models mainly use the historical values
of temperature to predict future values, while the ML models we used
earlier use multiple features (humidity, pressure, etc.).

Therefore, this provides a "different approach" that is useful for
comparison in the research paper.
"""


from common import DATA_DIR, OUTPUT_DIR, load_weather, prepare_dirs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- DATA LOAD  ----
df = load_weather(DATA_DIR / "delhi_weather_raw.csv")
prepare_dirs(OUTPUT_DIR)

# We will forecast only 'temp_mean' (daily mean temperature).
series = df.set_index("date")["temp_mean"]

# ---- CHRONOLOGICAL 80/20 SPLIT (aligned with the other forecasters) ----

split_idx = int(len(series) * 0.8)
train = series.iloc[:split_idx]
test = series.iloc[split_idx:]
TEST_DAYS = len(test)
if len(series) <= TEST_DAYS + 30:
    raise ValueError("At least 31 training days plus a holdout are required for time-series models.")
print(f"Training days: {len(train)}, Testing days: {len(test)}")

results = []
predictions = {}

# =========================================================
# MODEL 1: ARIMA
# =========================================================
print("\n[1/3] Training ARIMA...")
# order=(p,d,q): p = autoregressive terms, d = differencing, q = moving average terms

# (5,1,2) is a common starting point for daily weather data

arima_model = ARIMA(train, order=(5, 1, 2))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=TEST_DAYS)
predictions["ARIMA"] = arima_pred.values

mae = mean_absolute_error(test, arima_pred)
rmse = np.sqrt(mean_squared_error(test, arima_pred))
r2 = r2_score(test, arima_pred)
results.append({"Model": "ARIMA", "MAE": mae, "RMSE": rmse, "R2": r2})
print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

# =========================================================
# MODEL 2: SARIMA (Seasonal ARIMA)
# =========================================================
print("\n[2/3] Training SARIMA (seasonal_order ke saath)...")
# seasonal_order=(P,D,Q,s): s=seasonal period
# NOTE: s=365 (yearly/monsoon cycle) can be the most accurate, but it is

# very heavy and slow to train on this dataset size. When s=30 was tried,

# convergence failed (negative R², which indicates overfitting or a poor fit).

# s=7 (weekly) was the most stable for this dataset. This is a REAL trade-off

# that you can discuss in your research paper:

# "Seasonal period selection vs. computational cost".

sarima_model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                         enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima_model.fit(disp=False, maxiter=200)
sarima_pred = sarima_fit.forecast(steps=TEST_DAYS)
predictions["SARIMA"] = sarima_pred.values

mae = mean_absolute_error(test, sarima_pred)
rmse = np.sqrt(mean_squared_error(test, sarima_pred))
r2 = r2_score(test, sarima_pred)
results.append({"Model": "SARIMA", "MAE": mae, "RMSE": rmse, "R2": r2})
print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

# =========================================================
# MODEL 3: Prophet
# =========================================================
print("\n[3/3] Training Prophet...")
# Prophet requires specific column names: 'ds' (date) and 'y' (value)

prophet_train = train.reset_index()
prophet_train.columns = ["ds", "y"]

prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(periods=TEST_DAYS)
forecast = prophet_model.predict(future)
prophet_pred = forecast["yhat"].iloc[-TEST_DAYS:].values
predictions["Prophet"] = prophet_pred

mae = mean_absolute_error(test, prophet_pred)
rmse = np.sqrt(mean_squared_error(test, prophet_pred))
r2 = r2_score(test, prophet_pred)
results.append({"Model": "Prophet", "MAE": mae, "RMSE": rmse, "R2": r2})
print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

# =========================================================
# RESULTS COMPARISON
# =========================================================
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n" + "=" * 60)
print("TIME-SERIES MODEL COMPARISON (sorted by RMSE)")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "timeseries_model_comparison.csv", index=False)

best_model = results_df.iloc[0]["Model"]
print(f"\n🏆 Best Time-Series Model: {best_model}")

# =========================================================
# VISUALIZATION
# =========================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(test.index, test.values, label="Actual", color="black", linewidth=2)
colors = {"ARIMA": "red", "SARIMA": "blue", "Prophet": "green"}
for name, preds in predictions.items():
    axes[0].plot(test.index, preds, label=name, color=colors[name], linewidth=1, alpha=0.7)
axes[0].set_title("Time-Series Models: Actual vs Predicted Temperature (Test period)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()

axes[1].bar(results_df["Model"], results_df["RMSE"], color="darkorange")
axes[1].set_title("Time-Series Model Comparison (RMSE - lower is better)")
axes[1].set_ylabel("RMSE (°C)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timeseries_results.png", dpi=150, bbox_inches="tight")
print(f"\nGraphs saved: {OUTPUT_DIR / 'timeseries_results.png'}")

print("""
NOTE FOR YOUR RESEARCH PAPER:

- ARIMA/SARIMA/Prophet use only the historical values of temperature
  (univariate), while the ML models from Part 01 (Random Forest, XGBoost)
  use multiple features (multivariate).
- Therefore, if the ML models achieve better or similar RMSE compared to
  the time-series models, this is an interesting finding:
  "How much improvement was achieved by using additional features
  (humidity, pressure, etc.)?"

""")
