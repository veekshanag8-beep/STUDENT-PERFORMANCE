"""Quantify the leakage from PostExamConfidence.

Runs the same 5-fold CV for the chosen model with and without the post-exam
column, to show how much it would (falsely) improve the scores.

Run:  python src/leakage_check.py
"""
import sys

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE
from features import FEATURES, OutOfRangeToNaN, load_training_data
from train import CV


def hgb_pipeline(cols):
    prep = ColumnTransformer([("num", Pipeline([
        ("range_check", OutOfRangeToNaN(cols)),
        ("impute", SimpleImputer(strategy="median")),
    ]), cols)])
    return Pipeline([("prep", prep),
                     ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_training_data()
    for label, cols in [("without PostExamConfidence (used)", FEATURES),
                        ("WITH PostExamConfidence (leaky)", FEATURES + ["PostExamConfidence"])]:
        rmse = -cross_val_score(hgb_pipeline(cols), X, y, cv=CV,
                                scoring="neg_root_mean_squared_error")
        print(f"{label:36} RMSE {rmse.mean():.2f} ± {rmse.std():.2f}")


if __name__ == "__main__":
    main()
