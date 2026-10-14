"""Part 02 - Step 1: Multivariate LSTM/GRU (all weather variables -> next-day mean temp)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "delhi_weather_raw.csv"
WINDOW = 14          # last 14 days -> predict tomorrow
EPOCHS = 60

df = pd.read_csv(RAW)
date_col = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)
# Do not interpolate across future rows before splitting. Forward-fill only
# carries information already available at each date; leading gaps are removed.
feat = df.select_dtypes("number").ffill().dropna()

# target column: mean temperature (change here if your column name differs)
target = next(c for c in feat.columns if "temp" in c.lower() and "mean" in c.lower())
feat["doy_sin"] = np.sin(2 * np.pi * feat.index.dayofyear / 365.25)
feat["doy_cos"] = np.cos(2 * np.pi * feat.index.dayofyear / 365.25)
cols = list(feat.columns)
print("Target:", target, "| Features:", cols)

n = len(feat)
split = int(n * 0.8)
sx, sy = MinMaxScaler().fit(feat.iloc[:split]), MinMaxScaler().fit(feat[[target]].iloc[:split])
Xs, ys = sx.transform(feat), sy.transform(feat[[target]])

X, y, idx = [], [], []
for i in range(WINDOW - 1, n - 1):
    X.append(Xs[i - WINDOW + 1: i + 1]); y.append(ys[i + 1]); idx.append(i + 1)
X, y, idx = np.array(X), np.array(y), np.array(idx)
tr, te = idx < split, idx >= split

def build(kind):
    L = layers.LSTM if kind == "LSTM" else layers.GRU
    m = models.Sequential([layers.Input((WINDOW, X.shape[2])), L(64, return_sequences=True),
                           layers.Dropout(0.2), L(32), layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m

results, best = [], (-9, None, None)
for kind in ["LSTM", "GRU"]:
    m = build(kind)
    m.fit(X[tr], y[tr], validation_split=0.1, epochs=EPOCHS, batch_size=32, verbose=0,
          callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    pred = sy.inverse_transform(m.predict(X[te], verbose=0)).ravel()
    true = sy.inverse_transform(y[te]).ravel()
    r2 = r2_score(true, pred)
    results.append({"model": f"Multivariate {kind}", "R2": r2,
                    "MAE": mean_absolute_error(true, pred),
                    "RMSE": mean_squared_error(true, pred) ** 0.5})
    print(results[-1])
    if r2 > best[0]:
        best = (r2, m, (true, pred, kind))

(ROOT / "models").mkdir(exist_ok=True); (ROOT / "outputs").mkdir(exist_ok=True)
best[1].save(ROOT / "models" / "best_multivariate.keras")
joblib.dump({"sx": sx, "sy": sy}, ROOT / "models" / "scalers.joblib")
json.dump({"cols": cols, "target": target, "window": WINDOW, "kind": best[2][2]},
          open(ROOT / "models" / "meta.json", "w"))
pd.DataFrame(results).to_csv(ROOT / "outputs" / "multivariate_dl_comparison.csv", index=False)

plt.figure(figsize=(12, 4))
plt.plot(best[2][0], label="Actual"); plt.plot(best[2][1], label="Predicted")
plt.title(f"Best: Multivariate {best[2][2]} (R2={best[0]:.3f})"); plt.legend()
plt.savefig(ROOT / "outputs" / "multivariate_dl_results.png", dpi=150, bbox_inches="tight")
print("Saved model + results.")
