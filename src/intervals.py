"""Bonus - 90% prediction intervals via conformalized quantile regression (CQR).

1. Two extra gradient-boosting pipelines predict the 5th and 95th percentile of
   the score instead of the mean (same preprocessing + tuned hyperparameters,
   only the loss changes).
2. Raw quantile models are usually over-confident, so the interval is widened by
   a calibration margin: on inner-CV out-of-fold predictions, measure how far the
   true score falls outside [lower, upper] and take the 90% quantile of that.

Coverage is checked with the same outer 5-fold CV (the margin is learned inside
each outer training fold only): how often does the TRUE score fall inside the
interval for students the models never saw? Target: about 90%.

Run:  python src/intervals.py   (saves models/interval_models.joblib)
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, ROOT, TARGET, VALID_RANGES
from features import build_preprocessor, load_training_data
from train import CV

INTERVAL_PATH = ROOT / "models" / "interval_models.joblib"
LOWER_Q, UPPER_Q = 0.05, 0.95


def quantile_pipeline(q):
    params_file = ROOT / "reports" / "best_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else {}
    return Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("model", HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                                random_state=RANDOM_STATE, **params)),
    ])


def raw_interval(lower_model, upper_model, X):
    a, b = lower_model.predict(X), upper_model.predict(X)
    return np.minimum(a, b), np.maximum(a, b)  # guard against quantile crossing


def fit_calibrated(X, y):
    """Fit lower/upper quantile models and learn the conformal margin on inner CV."""
    inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = np.empty(len(y))
    for tr, va in inner.split(X):
        lo_m = quantile_pipeline(LOWER_Q).fit(X.iloc[tr], y.iloc[tr])
        hi_m = quantile_pipeline(UPPER_Q).fit(X.iloc[tr], y.iloc[tr])
        lower, upper = raw_interval(lo_m, hi_m, X.iloc[va])
        actual = y.iloc[va].values
        scores[va] = np.maximum(lower - actual, actual - upper)  # >0 means outside
    level = min(1.0, np.ceil((len(y) + 1) * (UPPER_Q - LOWER_Q)) / len(y))
    return {
        "lower": quantile_pipeline(LOWER_Q).fit(X, y),
        "upper": quantile_pipeline(UPPER_Q).fit(X, y),
        "margin": float(np.quantile(scores, level)),
    }


def predict_interval(models, X):
    lo, hi = VALID_RANGES[TARGET]
    lower, upper = raw_interval(models["lower"], models["upper"], X)
    lower, upper = lower - models["margin"], upper + models["margin"]
    return np.clip(lower, lo, hi), np.clip(upper, lo, hi)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_training_data()

    rows = []
    for train_idx, val_idx in CV.split(X):
        models = fit_calibrated(X.iloc[train_idx], y.iloc[train_idx])
        print(f"  fold margin: {models['margin']:+.2f} marks")
        lower, upper = predict_interval(models, X.iloc[val_idx])
        rows.append(pd.DataFrame({"actual": y.iloc[val_idx].values,
                                  "lower": lower, "upper": upper}))
    oof = pd.concat(rows, ignore_index=True)
    oof["inside"] = oof["actual"].between(oof["lower"], oof["upper"])
    oof["width"] = oof["upper"] - oof["lower"]

    nominal = int(round((UPPER_Q - LOWER_Q) * 100))
    print(f"Nominal coverage: {nominal}%   (out-of-fold, 5-fold CV)")
    print(f"Actual coverage:  {oof['inside'].mean():.1%}")
    print(f"Average width:    {oof['width'].mean():.1f} marks (median {oof['width'].median():.1f})")
    low = oof["actual"] < 50
    print(f"Coverage for students scoring < 50: {oof.loc[low, 'inside'].mean():.1%} (n={low.sum()})")
    print(f"Coverage for students scoring >= 50: {oof.loc[~low, 'inside'].mean():.1%}")

    pd.DataFrame([{
        "nominal_coverage": nominal / 100,
        "oof_coverage": round(oof["inside"].mean(), 4),
        "mean_width": round(oof["width"].mean(), 2),
        "coverage_below_50": round(oof.loc[low, "inside"].mean(), 4),
        "coverage_50_plus": round(oof.loc[~low, "inside"].mean(), 4),
    }]).to_csv(ROOT / "reports" / "interval_coverage.csv", index=False)

    final = fit_calibrated(X, y)
    final["coverage"] = f"{oof['inside'].mean():.0%}"
    joblib.dump(final, INTERVAL_PATH)
    print(f"\nFinal margin: {final['margin']:+.2f} marks. Saved {INTERVAL_PATH}")


if __name__ == "__main__":
    main()
