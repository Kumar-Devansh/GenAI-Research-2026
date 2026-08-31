"""
STEP 7: DEEP LEARNING MODELS (LSTM & GRU)
=========================================

Now we will train sequence-based deep learning models.

DIFFERENCE FROM THE ML MODELS IN PART 01:

* ML models (such as Random Forest) look at a SINGLE ROW of data.
* LSTM/GRU models look at a SEQUENCE (such as the previous 30 days)
  and then predict the next day.

This technique is called "sliding window" or "sequence-to-one" forecasting.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

# ---- ROBUST PATH HANDLING ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")

# ---- DATA LOAD ----
df = pd.read_csv(os.path.join(DATA_DIR, "delhi_weather_raw.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
series = df["temp_mean"].values.reshape(-1, 1)

# ---- SCALING (Neural networks perform better when data is scaled to the 0-1 range) ----

scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# ---- CREATE SEQUENCES (Sliding Window) ----

# WINDOW_SIZE=30 means: Use the previous 30 days to predict the 31st day.

WINDOW_SIZE = 30

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(series_scaled, WINDOW_SIZE)

# LSTM/GRU requires input shape: (samples, timesteps, features)

X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Total sequences: {X.shape[0]}, Window size: {WINDOW_SIZE}")

# ---- TRAIN-TEST SPLIT (while maintaining the time-series order) ----

split_idx = int(len(X) * 0.85)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training sequences: {len(X_train)}, Testing sequences: {len(X_test)}")

results = []
predictions = {}

# =========================================================
# MODEL 1: LSTM
# =========================================================
print("\n[1/2] Training LSTM...")
lstm_model = Sequential([
    LSTM(64, activation="tanh", return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)
print(f"  Training stopped at epoch {len(lstm_history.history['loss'])} (early stopping)")

lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
# Convert the scaled values back to the original units (°C)

lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
predictions["LSTM"] = lstm_pred

mae = mean_absolute_error(y_test_actual, lstm_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
r2 = r2_score(y_test_actual, lstm_pred)
results.append({"Model": "LSTM", "MAE": mae, "RMSE": rmse, "R2": r2})
print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

# =========================================================
# MODEL 2: GRU
# =========================================================
print("\n[2/2] Training GRU...")
gru_model = Sequential([
    GRU(64, activation="tanh", return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    GRU(32, activation="tanh"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
gru_model.compile(optimizer="adam", loss="mse")

gru_history = gru_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)
print(f"  Training stopped at epoch {len(gru_history.history['loss'])} (early stopping)")

gru_pred_scaled = gru_model.predict(X_test, verbose=0)
gru_pred = scaler.inverse_transform(gru_pred_scaled).flatten()
predictions["GRU"] = gru_pred

mae = mean_absolute_error(y_test_actual, gru_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, gru_pred))
r2 = r2_score(y_test_actual, gru_pred)
results.append({"Model": "GRU", "MAE": mae, "RMSE": rmse, "R2": r2})
print(f"  MAE: {mae:.3f}°C | RMSE: {rmse:.3f}°C | R²: {r2:.3f}")

gru_model.save(os.path.join(MODEL_DIR, "gru_model.keras"))

# =========================================================
# RESULTS
# =========================================================
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n" + "=" * 60)
print("DEEP LEARNING MODEL COMPARISON (sorted by RMSE)")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "deep_learning_comparison.csv"), index=False)

# =========================================================
# VISUALIZATION
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# 1. Actual vs Predicted
test_dates = df["date"].iloc[-len(y_test_actual):].values
axes[0].plot(test_dates, y_test_actual, label="Actual", color="black", linewidth=1.5)
axes[0].plot(test_dates, predictions["LSTM"], label="LSTM", color="crimson", alpha=0.8)
axes[0].plot(test_dates, predictions["GRU"], label="GRU", color="royalblue", alpha=0.8)
axes[0].set_title("Deep Learning Models: Actual vs Predicted Temperature")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()

# 2. Training loss curves (important for showing model actually learned)
axes[1].plot(lstm_history.history["loss"], label="LSTM - Training Loss")
axes[1].plot(lstm_history.history["val_loss"], label="LSTM - Validation Loss", linestyle="--")
axes[1].plot(gru_history.history["loss"], label="GRU - Training Loss")
axes[1].plot(gru_history.history["val_loss"], label="GRU - Validation Loss", linestyle="--")
axes[1].set_title("Training Loss Curves (MSE, scaled units)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

# 3. Model comparison bar chart
axes[2].bar(results_df["Model"], results_df["RMSE"], color="mediumpurple")
axes[2].set_title("Deep Learning Model Comparison (RMSE - lower is better)")
axes[2].set_ylabel("RMSE (°C)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "deep_learning_results.png"), dpi=120)
print(f"\nGraphs saved: {os.path.join(OUTPUT_DIR, 'deep_learning_results.png')}")

print("""
NOTE FOR YOUR RESEARCH PAPER:

- LSTM/GRU models only use the historical sequence of temperature
  (similar to time-series models), but they can also learn
  NON-LINEAR patterns (which ARIMA/SARIMA cannot capture effectively).
- Compare their R² values with the ML models (0.94+) and
  time-series models (0.11-0.21).
- Early stopping is used to prevent overfitting.
  This is an important deep learning best practice that should
  be mentioned in the methodology section.
  """)

