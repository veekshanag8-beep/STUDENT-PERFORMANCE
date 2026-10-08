"""Stage 5 - Inference CLI.

Usage:
    python predict.py --input data/student_performance_test.csv --output submission.csv

Loads the saved pipeline (models/pipeline.joblib), scores every row of the input
CSV and writes ID,FinalExamScore. All cleaning (out-of-range -> NaN, imputation,
scaling, dropping unused columns) happens inside the saved pipeline, so the input
only needs the raw columns.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # the pipeline uses classes from src/features.py

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import ID_COL, MODEL_PATH, TARGET, VALID_RANGES  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Predict FinalExamScore for students in a CSV.")
    p.add_argument("--input", required=True, help="CSV with an ID column and the feature columns")
    p.add_argument("--output", required=True, help="where to write ID,FinalExamScore")
    p.add_argument("--model", default=str(MODEL_PATH), help="path to the saved pipeline")
    p.add_argument("--intervals", metavar="PATH",
                   help="optional: also write ID,FinalExamScore,Lower90,Upper90 to this file "
                        "(the --output file keeps the strict 2-column format)")
    return p.parse_args()


def load_input(path, features):
    df = pd.read_csv(path)
    missing = [c for c in [ID_COL, *features] if c not in df.columns]
    if missing:
        sys.exit(f"Error: input is missing required column(s): {missing}")

    for col in features:  # stray text like "n/a" becomes NaN and gets imputed
        numeric = pd.to_numeric(df[col], errors="coerce")
        n_bad = numeric.isna().sum() - df[col].isna().sum()
        if n_bad:
            print(f"Warning: {n_bad} non-numeric value(s) in {col} treated as missing")
        df[col] = numeric
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    bundle = joblib.load(args.model)
    pipeline, features = bundle["pipeline"], bundle["features"]

    df = load_input(args.input, features)
    pred = pipeline.predict(df)
    lo, hi = VALID_RANGES[TARGET]
    pred = np.clip(pred, lo, hi)  # exam scores can't go outside 0-100

    out = pd.DataFrame({ID_COL: df[ID_COL], TARGET: pred.round(2)})
    out.to_csv(args.output, index=False, float_format="%.2f")
    print(f"Model: {bundle['model_name']} (CV RMSE {bundle['cv_rmse']})")
    print(f"Wrote {len(out)} predictions to {args.output}")

    if args.intervals:
        from intervals import INTERVAL_PATH, predict_interval
        interval_models = joblib.load(INTERVAL_PATH)
        lower, upper = predict_interval(interval_models, df)
        out.assign(Lower90=lower.round(2), Upper90=upper.round(2)).to_csv(
            args.intervals, index=False, float_format="%.2f")
        print(f"Wrote 90% prediction intervals (CV coverage {interval_models['coverage']}) "
              f"to {args.intervals}")


if __name__ == "__main__":
    main()
