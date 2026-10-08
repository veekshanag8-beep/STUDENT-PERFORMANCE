"""Stage 4 - Residual / error analysis of the chosen model.

Uses OUT-OF-FOLD predictions: each student is predicted by the model trained on
the other 4 folds, so these errors are honest (the model never saw that student).

Run:  python src/error_analysis.py
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from config import FIG_DIR, MODEL_PATH, ROOT, VALID_RANGES
from features import FEATURES, load_training_data
from train import CV, make_models


def had_bad_input(X):
    """True where any kept feature was missing or outside its valid range."""
    bad = X[FEATURES].isna()
    for col in FEATURES:
        lo, hi = VALID_RANGES[col]
        bad[col] |= (X[col] < lo) | (X[col] > hi)
    return bad.any(axis=1)


def segment_table(df, by):
    """n, RMSE, MAE and bias (mean of actual - predicted) for each group."""
    g = df.groupby(by, observed=True)["residual"]
    return pd.DataFrame({
        "n": g.size(),
        "RMSE": g.apply(lambda r: np.sqrt((r ** 2).mean())),
        "MAE": g.apply(lambda r: r.abs().mean()),
        "bias": g.mean(),  # >0: model under-predicts, <0: over-predicts
    }).round(2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_training_data()
    model_name = joblib.load(MODEL_PATH)["model_name"]
    pred = cross_val_predict(make_models()[model_name], X, y, cv=CV)

    df = X.copy()
    df["actual"] = y.values
    df["predicted"] = pred
    df["residual"] = df["actual"] - df["predicted"]
    overall_rmse = np.sqrt((df["residual"] ** 2).mean())
    print(f"Model: {model_name}   out-of-fold RMSE: {overall_rmse:.2f}\n")

    # --- Segments -------------------------------------------------------------
    df["actual_band"] = pd.cut(df["actual"], [0, 50, 65, 80, 95, 100.01], right=False,
                               labels=["<50", "50-65", "65-80", "80-95", "95-100"])
    df["backlogs"] = pd.cut(df["PreviousBacklogs"], [-np.inf, 0, 1, 2, np.inf],
                            labels=["0", "1", "2", "3+"])
    df["attendance"] = pd.cut(df["AttendancePercentage"], [-np.inf, 60, 75, 90, np.inf],
                              labels=["<60", "60-75", "75-90", "90+"])
    df["input_quality"] = np.where(had_bad_input(X), "missing/invalid input", "clean input")

    tables = {
        "Actual score band": segment_table(df, "actual_band"),
        "Previous backlogs": segment_table(df, "backlogs"),
        "Attendance %": segment_table(df, "attendance"),
        "Input quality": segment_table(df, "input_quality"),
    }
    for title, t in tables.items():
        print(f"--- {title} ---\n{t.to_string()}\n")
    print(f"Students scoring exactly 100: {(df['actual'] == 100).sum()}")

    worst = df.reindex(df["residual"].abs().sort_values(ascending=False).index).head(10)
    print("\n--- 10 largest errors ---")
    print(worst[["ID", *FEATURES, "actual", "predicted", "residual"]].round(1).to_string(index=False))

    out = ROOT / "reports"
    df[["ID", "actual", "predicted", "residual"]].round(3).to_csv(out / "oof_predictions.csv", index=False)
    pd.concat(tables, names=["segment_type", "segment"]).to_csv(out / "error_segments.csv")

    # --- Plots ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.scatter(df["predicted"], df["residual"], s=12, alpha=0.5, color="#4C72B0")
    ax.axhline(0, color="black", lw=1)
    ax.set(title="Residuals vs predicted", xlabel="predicted score",
           ylabel="residual (actual - predicted)")

    ax = axes[1]
    ax.scatter(df["actual"], df["predicted"], s=12, alpha=0.5, color="#4C72B0")
    lims = [df[["actual", "predicted"]].min().min() - 2, 102]
    ax.plot(lims, lims, color="red", ls="--", lw=1, label="perfect prediction")
    ax.set(title="Predicted vs actual", xlabel="actual score", ylabel="predicted score",
           xlim=lims, ylim=lims)
    ax.legend()

    ax = axes[2]
    band = tables["Actual score band"]
    bars = ax.bar(band.index.astype(str), band["RMSE"], color="#4C72B0")
    ax.axhline(overall_rmse, color="red", ls="--", lw=1, label=f"overall RMSE {overall_rmse:.2f}")
    ax.bar_label(bars, labels=[f"n={n}" for n in band["n"]], fontsize=8)
    ax.set(title="Error by actual score band", xlabel="actual score", ylabel="RMSE")
    ax.legend()

    fig.suptitle(f"{model_name} - out-of-fold error analysis (5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=120)
    print(f"\nSaved {FIG_DIR / 'residuals.png'} and reports/error_segments.csv")


if __name__ == "__main__":
    main()
