"""Chronological two-stage next-day rainfall experiment (experimental)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_rainfall_frame(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    required = {"date", "rainfall"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing rainfall fields: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "rainfall"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 dated observations are required for lagged features.")
    frame["target_rainfall_next_day"] = frame["rainfall"].shift(-1)
    frame["target_date"] = frame["date"].shift(-1)
    frame["day_sin"] = np.sin(2 * np.pi * frame["date"].dt.dayofyear / 365.25)
    frame["day_cos"] = np.cos(2 * np.pi * frame["date"].dt.dayofyear / 365.25)
    for lag in (1, 2, 3, 7):
        for col in ("rainfall", "temp_mean", "humidity", "pressure", "cloud_cover"):
            if col in frame:
                frame[f"{col}_lag{lag}"] = frame[col].shift(lag)
    frame["rain_7d_sum"] = frame["rainfall"].rolling(7, min_periods=7).sum()
    frame = frame.dropna().reset_index(drop=True)
    excluded = {"date", "target_date", "target_rainfall_next_day"}
    numeric = frame.select_dtypes(include="number").drop(columns=list(excluded & set(frame.columns)), errors="ignore")
    # Open-Meteo archive files currently contain duplicate rain/rainfall columns.
    numeric = numeric.loc[:, ~numeric.T.duplicated()]
    features = list(numeric.columns)
    if not features:
        raise ValueError("No numeric predictors found.")
    return frame, features


def evaluate(path: Path, output_dir: Path) -> dict:
    """Use first 70% train, next 15% validation, last 15% test chronologically."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 mean_absolute_error, mean_squared_error, precision_score, recall_score)
    frame, features = build_rainfall_frame(path)
    n = len(frame)
    if n < 30:
        raise ValueError("At least 30 complete lagged rows are required for train/validation/test evaluation.")
    train_end, val_end = int(n * .70), int(n * .85)
    if train_end < 10 or val_end <= train_end or val_end >= n:
        raise ValueError("Unable to create non-empty chronological splits.")
    x = frame[features].astype(float)
    y = frame["target_rainfall_next_day"].astype(float)
    wet_train = y.iloc[:train_end] > 0
    clf = (HistGradientBoostingClassifier(max_iter=100, learning_rate=.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=42)
           if wet_train.nunique() == 2 else DummyClassifier(strategy="prior"))
    clf.fit(x.iloc[:train_end], wet_train.astype(int))
    val_probability = clf.predict_proba(x.iloc[train_end:val_end])[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(val_end-train_end)
    thresholds = np.linspace(.1, .9, 17)
    threshold = max(thresholds, key=lambda t: f1_score((y.iloc[train_end:val_end] > 0), val_probability >= t, zero_division=0))
    test_prob = clf.predict_proba(x.iloc[val_end:])[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(n-val_end)
    test_y = y.iloc[val_end:].to_numpy()
    truth_wet = test_y > 0
    pred_wet = test_prob >= threshold
    # Conditional wet-day amount: fit only on wet training examples, falling back
    # to a training-period median if the rainfall class is absent.
    wet_x = x.iloc[:train_end].loc[wet_train]
    wet_y = y.iloc[:train_end].loc[wet_train]
    amount_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=.06, max_leaf_nodes=15,
                                                 l2_regularization=1.0, random_state=42)
    if len(wet_y) >= 5:
        amount_model.fit(wet_x, wet_y)
        conditional = np.maximum(0, amount_model.predict(x.iloc[val_end:]))
    else:
        conditional = np.full(n-val_end, float(wet_y.median()) if len(wet_y) else 0.)
    amount_pred = np.where(pred_wet, conditional, 0.)
    baseline_clf = DummyClassifier(strategy="prior").fit(x.iloc[:train_end], wet_train.astype(int))
    base_probability = baseline_clf.predict_proba(x.iloc[val_end:])[:, list(baseline_clf.classes_).index(1)] if 1 in baseline_clf.classes_ else np.zeros(n-val_end)
    base_wet = base_probability >= .5
    baseline_amount = DummyRegressor(strategy="median").fit(x.iloc[:train_end], y.iloc[:train_end])
    baseline_pred = np.maximum(0, baseline_amount.predict(x.iloc[val_end:]))
    conf = confusion_matrix(truth_wet, pred_wet, labels=[0, 1]).tolist()
    metrics = {
        "experiment": "experimental next-day rainfall; Open-Meteo archive labels are model-derived weather estimates",
        "split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, n],
                  "train_last_target_date": str(frame.target_date.iloc[train_end-1].date()),
                  "test_first_target_date": str(frame.target_date.iloc[val_end].date()),
                  "test_last_target_date": str(frame.target_date.iloc[-1].date())},
        "wet_threshold_mm": 0.0, "decision_threshold_selected_on_validation": float(threshold),
        "classification": {"precision": float(precision_score(truth_wet, pred_wet, zero_division=0)),
                           "recall": float(recall_score(truth_wet, pred_wet, zero_division=0)),
                           "f1": float(f1_score(truth_wet, pred_wet, zero_division=0)),
                           "pr_auc": float(average_precision_score(truth_wet, test_prob)) if truth_wet.any() else None,
                           "confusion_matrix_labels_0_1": conf},
        "baseline": {"classifier": {"precision": float(precision_score(truth_wet, base_wet, zero_division=0)),
                                      "recall": float(recall_score(truth_wet, base_wet, zero_division=0)),
                                      "f1": float(f1_score(truth_wet, base_wet, zero_division=0)),
                                      "pr_auc": float(average_precision_score(truth_wet, base_probability)) if truth_wet.any() else None},
                     "median_amount_all_test_days_mm": {"mae": float(mean_absolute_error(test_y, baseline_pred)),
                                                        "rmse": float(mean_squared_error(test_y, baseline_pred) ** .5)}},
        "amount_all_test_days_mm": {"mae": float(mean_absolute_error(test_y, amount_pred)),
                                    "rmse": float(mean_squared_error(test_y, amount_pred) ** .5)},
        "amount_wet_test_days_mm": ({"mae": float(mean_absolute_error(test_y[truth_wet], amount_pred[truth_wet])),
                                     "rmse": float(mean_squared_error(test_y[truth_wet], amount_pred[truth_wet]) ** .5)} if truth_wet.any() else None),
        "test_wet_fraction": float(truth_wet.mean()), "n_rows": n, "features": features,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rainfall_experiment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"target_date": frame.target_date.iloc[val_end:], "actual_mm": test_y,
                  "wet_probability": test_prob, "predicted_mm": amount_pred}).to_csv(output_dir / "rainfall_experiment_predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    test_dates = pd.to_datetime(frame.target_date.iloc[val_end:])
    axes[0].plot(test_dates, test_y, color="#247ba0", linewidth=1, label="Archive target")
    axes[0].plot(test_dates, amount_pred, color="#f18f01", linewidth=1, alpha=.85, label="Two-stage estimate")
    axes[0].set_ylabel("Rainfall (mm)")
    axes[0].set_title("Next-day rainfall · chronological test period")
    axes[0].legend(frameon=False)
    axes[1].plot(test_dates, test_prob, color="#3c8d70", linewidth=1, label="Wet-day probability")
    axes[1].axhline(threshold, color="#bd4f6c", linestyle="--", label=f"Validation threshold {threshold:.2f}")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "rainfall_experiment_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    result = evaluate(ROOT / "data" / "delhi_weather_raw.csv", ROOT / "outputs")
    print(json.dumps(result, indent=2))
