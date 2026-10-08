# Student Performance Prediction

Predict a student's `FinalExamScore` from information available **before** the exam,
using a scikit-learn `Pipeline` / `ColumnTransformer`, cross-validated model comparison,
a packaged CLI (`predict.py`) and a Streamlit demo (`app.py`).

Full write-up: [REPORT.md](REPORT.md)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

```bash
python src/audit.py                     # data audit -> reports/figures
python src/train.py                     # CV comparison + saves models/pipeline.joblib
python src/leakage_check.py             # CV with vs without the leaky PostExamConfidence column
python src/error_analysis.py            # residual plots + per-segment errors
python predict.py --input data/student_performance_test.csv --output submission.csv
streamlit run app.py                    # interactive demo
```

## Project layout

```
data/        raw CSVs (train + test)
src/         audit, feature/pipeline definitions, training, error analysis
models/      saved fitted pipeline
reports/     figures and tables used in REPORT.md
predict.py   inference CLI
app.py       Streamlit demo
```
