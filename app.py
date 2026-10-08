"""Stage 6 - Streamlit demo.

Run:  streamlit run app.py
Loads the same saved pipeline as predict.py and scores one student from the sliders.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # the pipeline uses classes from src/features.py

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import ID_COL, MODEL_PATH, TARGET, VALID_RANGES  # noqa: E402
from intervals import INTERVAL_PATH, predict_interval  # noqa: E402

# label, step, help text for each feature
INPUTS = {
    "PreviousExamScore": ("Previous exam score", 1.0, "Score on an earlier exam (0-100)"),
    "AttendancePercentage": ("Attendance (%)", 0.5, "Share of classes attended this term"),
    "AssignmentsCompleted": ("Assignments completed (%)", 1.0, "Share of coursework submitted"),
    "StudyHours": ("Study hours per day", 0.1, "Average self-study per day"),
    "SleepHours": ("Sleep hours per night", 0.1, "Average sleep per night"),
    "ClassParticipation": ("Class participation (0-10)", 0.1, "Instructor's participation rating"),
    "ExtracurricularHours": ("Extracurricular hours per week", 0.5, "Sports, clubs, etc."),
    "PreviousBacklogs": ("Previous backlogs", 1.0, "Number of earlier failed courses"),
}


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    # Medians the imputer learned from training data -> sensible slider defaults
    imputer = bundle["pipeline"].named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    bundle["defaults"] = dict(zip(bundle["features"], imputer.statistics_))
    bundle["intervals"] = joblib.load(INTERVAL_PATH) if INTERVAL_PATH.exists() else None
    return bundle


def predict(bundle, df):
    lo, hi = VALID_RANGES[TARGET]
    return np.clip(bundle["pipeline"].predict(df), lo, hi)


st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓")
bundle = load_model()
rmse = float(bundle["cv_rmse"].split()[0])

st.title("🎓 Final Exam Score Predictor")
st.caption(f"Model: {bundle['model_name']} · cross-validated RMSE {bundle['cv_rmse']} marks · "
           "uses only information available before the exam")

col_in, col_out = st.columns([3, 2], gap="large")
with col_in:
    st.subheader("Student details")
    values = {}
    for feat in bundle["features"]:
        label, step, help_text = INPUTS[feat]
        lo, hi = VALID_RANGES[feat]
        default = round(float(bundle["defaults"][feat]) / step) * step
        if step == 1.0:
            values[feat] = st.slider(label, int(lo), int(hi), int(default), 1, help=help_text)
        else:
            values[feat] = st.slider(label, float(lo), float(hi), float(default), step,
                                     help=help_text)

with col_out:
    score = predict(bundle, pd.DataFrame([values]))[0]
    st.subheader("Prediction")
    st.metric("Predicted final exam score", f"{score:.1f} / 100")
    st.progress(score / 100)
    if bundle["intervals"]:
        lower, upper = predict_interval(bundle["intervals"], pd.DataFrame([values]))
        st.write(f"**90% prediction interval: {lower[0]:.0f} – {upper[0]:.0f}**  \n"
                 f"In cross-validation the true score fell inside this range "
                 f"{bundle['intervals']['coverage']} of the time.")
    else:
        st.write(f"Typical error: about **±{rmse:.0f} marks** "
                 f"(likely range {max(score - rmse, 0):.0f}–{min(score + rmse, 100):.0f}).")
    if score < 60:
        st.warning("Predicted low score. Note: the model tends to **over-estimate** students who "
                   "end up below 50, so the real risk may be higher than shown.")
    st.info("Defaults are the training-set medians. This is a decision-support estimate, "
            "not a grade, and should not be used to make decisions about a student on its own.")

with st.expander("Score a whole CSV file"):
    up = st.file_uploader("CSV with an ID column and the feature columns", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in [ID_COL, *bundle["features"]] if c not in df.columns]
        if missing:
            st.error(f"Missing column(s): {missing}")
        else:
            out = pd.DataFrame({ID_COL: df[ID_COL], TARGET: predict(bundle, df).round(2)})
            st.dataframe(out, hide_index=True, height=240)
            st.download_button("Download submission.csv", out.to_csv(index=False),
                               "submission.csv", "text/csv")
