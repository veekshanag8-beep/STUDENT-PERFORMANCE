"""Bonus - Permutation feature importance.

For each CV fold: fit the final pipeline on the training part, then shuffle one
feature at a time in the VALIDATION part and measure how much RMSE gets worse.
Big increase = the model relies on that feature. Measured on held-out data, so it
reflects what generalises, not what was memorised.

Run:  python src/importance.py
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from config import FIG_DIR, MODEL_PATH, RANDOM_STATE, ROOT
from features import FEATURES, load_training_data
from train import CV

N_REPEATS = 10


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_training_data()
    X = X[FEATURES]
    bundle = joblib.load(MODEL_PATH)

    per_fold = []
    for train_idx, val_idx in CV.split(X):
        model = clone(bundle["pipeline"]).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            model, X.iloc[val_idx], y.iloc[val_idx], n_repeats=N_REPEATS,
            scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
        )
        per_fold.append(result.importances)  # shape (features, repeats)

    all_runs = np.concatenate(per_fold, axis=1)  # 5 folds x 10 repeats per feature
    imp = pd.DataFrame({
        "feature": FEATURES,
        "rmse_increase_mean": all_runs.mean(axis=1),
        "rmse_increase_std": all_runs.std(axis=1),
    }).sort_values("rmse_increase_mean", ascending=False)
    print(f"Model: {bundle['model_name']}\n"
          f"RMSE increase when a feature is shuffled ({CV.get_n_splits()} folds x {N_REPEATS} repeats):")
    print(imp.round(3).to_string(index=False))
    imp.round(4).to_csv(ROOT / "reports" / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    order = imp.iloc[::-1]
    ax.barh(order["feature"], order["rmse_increase_mean"], xerr=order["rmse_increase_std"],
            color="#4C72B0", capsize=3)
    ax.set(title="Permutation importance (held-out folds)",
           xlabel="increase in RMSE when feature is shuffled (marks)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=120)
    print(f"\nSaved {FIG_DIR / 'permutation_importance.png'}")


if __name__ == "__main__":
    main()
