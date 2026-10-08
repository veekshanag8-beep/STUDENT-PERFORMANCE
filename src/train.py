"""Stage 3 - Cross-validated comparison of 4 models.

Every model is a full Pipeline(preprocessor -> regressor), so cross_validate()
re-fits the imputer/scaler inside each training fold. The validation fold is
never seen while fitting -> no leakage.

Run:  python src/train.py
"""
import sys

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, ROOT
from features import build_preprocessor, load_training_data

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}


def make_models():
    """The 4 candidates. Trees don't need scaling, so they skip it."""
    return {
        "Mean baseline": Pipeline([
            ("prep", build_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge (linear)": Pipeline([
            ("prep", build_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]),
    }


def compare_models(X, y):
    rows = []
    for name, pipe in make_models().items():
        scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            if metric != "R2":  # sklearn returns errors as negatives
                vals = -vals
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
        print(f"  {name:24} RMSE {row['RMSE_mean']:.2f} ± {row['RMSE_std']:.2f}")
    return pd.DataFrame(rows).sort_values("RMSE_mean").reset_index(drop=True)


def to_markdown(results):
    lines = ["| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |",
             "|---|---|---|---|"]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['model']} | {r['RMSE_mean']:.2f} ± {r['RMSE_std']:.2f} "
            f"| {r['MAE_mean']:.2f} ± {r['MAE_std']:.2f} "
            f"| {r['R2_mean']:.3f} ± {r['R2_std']:.3f} |"
        )
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # so ± and ² print on Windows consoles
    X, y = load_training_data()
    print(f"\nCV strategy: {CV.get_n_splits()}-fold KFold, shuffle=True, "
          f"random_state={RANDOM_STATE}. Primary metric: RMSE (lower is better)\n")
    results = compare_models(X, y)

    out_dir = ROOT / "reports"
    results.round(4).to_csv(out_dir / "cv_results.csv", index=False)
    table = to_markdown(results)
    (out_dir / "cv_results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nBest model: {results.loc[0, 'model']}")


if __name__ == "__main__":
    main()
