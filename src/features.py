"""Stage 2 - Feature selection, data loading and the preprocessing pipeline.

Everything that LEARNS from data (imputer medians, scaler means/stds) lives inside
the sklearn ColumnTransformer returned by build_preprocessor(), so during
cross-validation it is fitted on the training folds only.

Run:  python src/features.py   (prints the feature decisions + a sanity check)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ID_COL, TARGET, TRAIN_PATH, VALID_RANGES

# Feature justification: could this value realistically be known BEFORE the final exam?
FEATURE_DECISIONS = {
    "ID": ("drop", "Row identifier, carries no information about the student."),
    "StudyHours": ("keep", "Self-reported study habit during the term, known before the exam."),
    "AttendancePercentage": ("keep", "Recorded by the college throughout the term."),
    "PreviousExamScore": ("keep", "Result of an earlier exam, already known."),
    "AssignmentsCompleted": ("keep", "Coursework submitted during the term."),
    "SleepHours": ("keep", "Lifestyle habit that can be surveyed before the exam."),
    "ExtracurricularHours": ("keep", "Known before the exam; weak signal but harmless to keep."),
    "ClassParticipation": ("keep", "Instructor rating given during the term."),
    "PreviousBacklogs": ("keep", "Count of earlier failed courses, part of the academic record."),
    "PostExamConfidence": ("drop", "LEAKAGE: collected AFTER the exam, so it reflects how the "
                                   "exam went and is unavailable at prediction time."),
}

FEATURES = [c for c, (decision, _) in FEATURE_DECISIONS.items() if decision == "keep"]


class OutOfRangeToNaN(BaseEstimator, TransformerMixin):
    """Replace impossible values (e.g. StudyHours=99, SleepHours=-1) with NaN.

    Uses fixed domain rules from config.VALID_RANGES, NOT statistics from the data,
    so it learns nothing in fit(). The imputer that follows fills the NaNs.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns).astype(float)
        for col in self.columns:
            lo, hi = VALID_RANGES[col]
            X.loc[(X[col] < lo) | (X[col] > hi), col] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


def build_preprocessor(scale=True):
    """Clean -> impute -> (optionally) scale the kept features; drop every other column."""
    steps = [
        ("range_check", OutOfRangeToNaN(FEATURES)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:  # linear models need scaling; trees don't care, so it's optional
        steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [("num", Pipeline(steps), FEATURES)],
        remainder="drop",  # ID and PostExamConfidence never reach the model
    )


def load_training_data(path=TRAIN_PATH):
    """Load training data and remove rows that can't be learned from.

    These are fixed row-level rules (no statistics), so applying them before
    cross-validation does not leak information between folds.
    """
    df = pd.read_csv(path)
    n0 = len(df)

    lo, hi = VALID_RANGES[TARGET]
    df = df[df[TARGET].between(lo, hi)]
    n_bad_target = n0 - len(df)

    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COL])
    n_dupes = n0 - n_bad_target - len(df)

    print(f"Loaded {n0} rows: dropped {n_bad_target} invalid-target rows "
          f"and {n_dupes} duplicate rows -> {len(df)} rows")
    return df.drop(columns=[TARGET]), df[TARGET]


if __name__ == "__main__":
    print("Feature decisions:")
    for col, (decision, reason) in FEATURE_DECISIONS.items():
        print(f"  {decision.upper():4}  {col:22} {reason}")

    X, y = load_training_data()
    Xt = build_preprocessor().fit_transform(X)
    print(f"\nPreprocessed shape: {Xt.shape}  (expected {len(FEATURES)} features)")
    print(f"NaNs after preprocessing: {np.isnan(Xt).sum()}")
    print(f"Feature means after scaling (should be ~0): {Xt.mean(axis=0).round(3)}")
