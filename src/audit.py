"""Stage 1 - Data audit.

Prints shape, missing values, bad records, duplicates and target statistics,
and saves audit figures to reports/figures/.

Run:  python src/audit.py
"""
import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DIR, ID_COL, TARGET, TEST_PATH, TRAIN_PATH, VALID_RANGES


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def count_out_of_range(df):
    rows = []
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df:
            continue
        s = df[col]
        rows.append({
            "column": col,
            "valid_range": f"[{lo}, {hi}]",
            "below": int((s < lo).sum()),
            "above": int((s > hi).sum()),
            "examples": sorted(s[(s < lo) | (s > hi)].unique().tolist())[:6],
        })
    return pd.DataFrame(rows)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    section("1. Shape and dtypes")
    print(f"train: {train.shape}   test: {test.shape}")
    print(train.dtypes.to_string())
    print(f"\nColumns in train but not in test: {set(train.columns) - set(test.columns)}")

    section("2. Missing values (count / %)")
    miss = pd.DataFrame({
        "train_missing": train.isna().sum(),
        "train_%": (train.isna().mean() * 100).round(1),
        "test_missing": test.isna().sum(),
    })
    print(miss.to_string())

    section("3. Summary statistics")
    print(train.drop(columns=ID_COL).describe().T.round(2).to_string())

    section("4. Bad records (outside valid ranges)")
    print("train:")
    print(count_out_of_range(train).to_string(index=False))
    print("\ntest:")
    print(count_out_of_range(test).to_string(index=False))

    section("5. Duplicates")
    feature_cols = [c for c in train.columns if c != ID_COL]
    print(f"duplicate IDs:                  {train[ID_COL].duplicated().sum()}")
    print(f"duplicate rows (ignoring ID):   {train.duplicated(subset=feature_cols).sum()}")

    section("6. Target distribution")
    y = train[TARGET]
    lo, hi = VALID_RANGES[TARGET]
    print(y.describe().round(2).to_string())
    print(f"\ntarget outside [{lo}, {hi}]: {((y < lo) | (y > hi)).sum()} rows")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(lo, color="red", ls="--", lw=1)
    ax.axvline(hi, color="red", ls="--", lw=1, label="valid range")
    ax.set(title="FinalExamScore distribution (raw)", xlabel="score", ylabel="students")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png", dpi=120)

    section("7. Correlation with target (valid-target rows only)")
    valid = train[(y >= lo) & (y <= hi)]
    corr = valid.drop(columns=ID_COL).corr()[TARGET].drop(TARGET).sort_values()
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(corr.index, corr.values, color=["#C44E52" if v < 0 else "#4C72B0" for v in corr])
    ax.set(title="Pearson correlation with FinalExamScore", xlabel="correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_correlations.png", dpi=120)

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
