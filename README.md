# Student Performance Prediction

Predict a student's `FinalExamScore` from information available **before** the exam,
using a scikit-learn `Pipeline` / `ColumnTransformer`, cross-validated model comparison
with nested hyperparameter tuning, a packaged CLI (`predict.py`) and a Streamlit demo (`app.py`).

**Result:** nested 5-fold CV RMSE **6.71 ± 0.39** marks (mean baseline: 14.04), with
90% prediction intervals (92% CV coverage).

Full write-up: [REPORT.md](REPORT.md) · Live demo: **[https://student-score-predictor11.streamlit.app/](https://student-score-predictor11.streamlit.app/)**

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt    # inference + demo
pip install -r requirements-dev.txt  # + MLflow, for re-running training with tracking
```

## Usage

```bash
python src/audit.py                     # data audit -> reports/figures
python src/train.py                     # CV comparison + nested tuning + saves models/pipeline.joblib
python src/leakage_check.py             # CV with vs without the leaky PostExamConfidence column
python src/error_analysis.py            # residual plots + per-segment errors
python src/importance.py                # permutation feature importance
python src/intervals.py                 # conformalized 90% prediction intervals
python predict.py --input data/student_performance_test.csv --output submission.csv
python predict.py --input data/student_performance_test.csv --output submission.csv --intervals intervals.csv
streamlit run app.py                    # interactive demo
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse logged experiment runs
```

## Project layout

```
data/        raw CSVs (train + test)
src/         audit, feature/pipeline definitions, training, error analysis,
             importance, prediction intervals, leakage check
models/      saved fitted pipeline + interval models
reports/     figures and tables used in REPORT.md
predict.py   inference CLI
app.py       Streamlit demo
```
