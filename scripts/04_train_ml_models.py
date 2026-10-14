"""
STEP 4: MACHINE LEARNING MODEL TRAINING
=======================================

We will train and compare multiple ML models to predict
the next day's temperature:

1. Linear Regression   (baseline - simplest model)
2. Ridge Regression    (Linear Regression + regularization)
3. Random Forest       (tree-based ensemble)
4. Gradient Boosting   (boosting technique)
5. XGBoost             (industry-standard, fast & accurate)

Metrics: MAE, RMSE, R² (metrics mentioned in the PDF)
"""


from common import DATA_DIR, MODEL_DIR, OUTPUT_DIR, load_weather, prepare_dirs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# ---- DATA LOAD  ----
df = load_weather(DATA_DIR / "delhi_weather_features.csv", ("date", "target_temp_next_day", "target_rainfall_next_day"))
prepare_dirs(MODEL_DIR, OUTPUT_DIR)

# ---- SEPARATE FEATURES (X) AND TARGET (y) ----

# In this script, we are only predicting the "next day's temperature".

feature_cols = [c for c in df.columns if c not in
                 ["date", "target_temp_next_day", "target_rainfall_next_day"]]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df["target_temp_next_day"]
if X.isna().any().any() or y.isna().any():
    raise ValueError("Feature data contains missing or non-numeric values; rerun feature engineering.")

# ---- TRAIN-TEST SPLIT ----

# IMPORTANT: This is time-series data, so we will NOT use a RANDOM split!

# Otherwise, future data could "leak" into the training data.

# Therefore: first 80% (older data) for training, last 20% (recent data) for testing.

raw_dates = pd.to_datetime(pd.read_csv(DATA_DIR / "delhi_weather_raw.csv", usecols=["date"])["date"])
if len(raw_dates) < 5:
    raise ValueError("At least five daily observations are required for a chronological holdout.")
holdout_start = raw_dates.iloc[int(len(raw_dates) * 0.8)]
target_dates = df["date"] + pd.Timedelta(days=1)
test_mask = target_dates >= holdout_start
X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
y_train, y_test = y.loc[~test_mask], y.loc[test_mask]
dates_test = target_dates.loc[test_mask]
if X_train.empty or X_test.empty:
    raise ValueError("Unable to form non-empty chronological train/test sets from feature data.")

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---- FEATURE SCALING (Necessary for Linear/Ridge models) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- MODELS DEFINE  ----
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
}

results = []
predictions = {}

for name, model in models.items():
    print(f"\nTraining: {name}...")

    # Linear models scaled data pe better chalte hain, tree models ko farak nahi padta
    if "Regression" in name and "Random" not in name and "Gradient" not in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    predictions[name] = y_pred

    print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

    # Best model save karenge baad mein
    joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').lower()}.pkl")

# ---- RESULTS COMPARISON TABLE ----
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON (sorted by RMSE - lower is better)")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")

# ---- VISUALIZATION: Actual vs Predicted (best model) ----
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(dates_test.values, y_test.values, label="Actual", color="black", linewidth=1.5)
axes[0].plot(dates_test.values, predictions[best_model_name], label=f"Predicted ({best_model_name})",
             color="red", linewidth=1, alpha=0.8)
axes[0].set_title(f"Actual vs Predicted Temperature - {best_model_name}")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()

# ---- MODEL COMPARISON BAR CHART ----
axes[1].bar(results_df["Model"], results_df["RMSE"], color="teal")
axes[1].set_title("Model Comparison (RMSE - lower is better)")
axes[1].set_ylabel("RMSE (°C)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_results.png", dpi=150, bbox_inches="tight")
print(f"\nGraphs saved: {OUTPUT_DIR / 'model_results.png'}")
