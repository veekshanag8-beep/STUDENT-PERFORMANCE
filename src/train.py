"""Stage 3/4 - Cross-validated comparison of the models, then save the best one.

Every model is a full Pipeline(preprocessor -> regressor), so cross_validate()
re-fits the imputer/scaler inside each training fold. The validation fold is
never seen while fitting -> no leakage.

The tuned model is a RandomizedSearchCV wrapped around the pipeline. Running it
through cross_validate() gives NESTED CV: the search only sees the outer training
fold, so its reported score is an honest estimate.

Run:  python src/train.py
"""
import json
import os
import sys
import time

import joblib
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, RANDOM_STATE, ROOT
from features import FEATURES, build_preprocessor, load_training_data

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}

# Tuning compute budget: N_ITER random settings x INNER_CV folds per search
N_ITER = 30
INNER_CV = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
HGB_SEARCH_SPACE = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(100, 600),
    "model__max_leaf_nodes": randint(8, 64),
    "model__max_depth": [None, 3, 5, 8],
    "model__min_samples_leaf": randint(5, 60),
    "model__l2_regularization": loguniform(1e-3, 10),
}


def hgb_pipeline():
    return Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def make_models():
    """The candidates. Trees don't need scaling, so they skip it."""
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
        "Hist Gradient Boosting": hgb_pipeline(),
        "Hist Gradient Boosting (tuned)": RandomizedSearchCV(
            hgb_pipeline(), HGB_SEARCH_SPACE, n_iter=N_ITER, cv=INNER_CV,
            scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def start_tracking():
    """Experiment tracking with MLflow if installed (pip install -r requirements-dev.txt)."""
    try:
        import mlflow
    except ImportError:
        print("(MLflow not installed - skipping experiment tracking)")
        return None
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("student-performance")
    return mlflow


def log_run(mlflow, name, model, row):
    if mlflow is None:
        return
    estimator = model.estimator if isinstance(model, RandomizedSearchCV) else model
    params = {k.removeprefix("model__"): v for k, v in estimator.get_params().items()
              if k.startswith("model__")}
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": name, "cv": f"{CV.get_n_splits()}-fold KFold seed {RANDOM_STATE}",
                           "n_features": len(FEATURES), **params})
        if isinstance(model, RandomizedSearchCV):
            mlflow.log_params({"tuning": "RandomizedSearchCV (nested CV)", "n_iter": N_ITER,
                               "inner_cv": INNER_CV.get_n_splits()})
        mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model"})


def compare_models(X, y):
    mlflow = start_tracking()
    rows = []
    for name, model in make_models().items():
        start = time.perf_counter()
        scores = cross_validate(model, X, y, cv=CV, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            if metric != "R2":  # sklearn returns errors as negatives
                vals = -vals
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        row["cv_seconds"] = time.perf_counter() - start
        rows.append(row)
        log_run(mlflow, name, model, row)
        print(f"  {name:32} RMSE {row['RMSE_mean']:.2f} ± {row['RMSE_std']:.2f}"
              f"   ({row['cv_seconds']:.1f}s)")
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
          f"random_state={RANDOM_STATE}. Primary metric: RMSE (lower is better)")
    print(f"Tuning budget: {N_ITER} settings x {INNER_CV.get_n_splits()} inner folds x "
          f"{CV.get_n_splits()} outer folds = {N_ITER * INNER_CV.get_n_splits() * CV.get_n_splits()}"
          f" fits, on {os.cpu_count()} CPU threads\n")
    results = compare_models(X, y)

    out_dir = ROOT / "reports"
    results.round(4).to_csv(out_dir / "cv_results.csv", index=False)
    table = to_markdown(results)
    (out_dir / "cv_results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    # Refit the winner on ALL training rows and save it for predict.py / app.py
    best_name = results.loc[0, "model"]
    best = make_models()[best_name].fit(X, y)
    if isinstance(best, RandomizedSearchCV):  # final search on all rows -> keep its best pipeline
        params = {k.removeprefix("model__"): v for k, v in best.best_params_.items()}
        (out_dir / "best_params.json").write_text(json.dumps(params, indent=2, default=str))
        print(f"\nBest hyperparameters: {params}")
        best = best.best_estimator_

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": best,
        "model_name": best_name,
        "features": FEATURES,
        "cv_rmse": f"{results.loc[0, 'RMSE_mean']:.2f} ± {results.loc[0, 'RMSE_std']:.2f}",
        "sklearn_version": sklearn.__version__,
    }, MODEL_PATH)
    print(f"\nBest model: {best_name} -> refitted on all {len(X)} rows, saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
