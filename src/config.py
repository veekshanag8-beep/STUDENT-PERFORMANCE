"""Shared constants: file paths, column names, valid value ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "student_performance.csv"
TEST_PATH = DATA_DIR / "student_performance_test.csv"
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
FIG_DIR = ROOT / "reports" / "figures"

ID_COL = "ID"
TARGET = "FinalExamScore"
RANDOM_STATE = 42

# Physically / logically valid ranges. Values outside these are treated as bad records.
VALID_RANGES = {
    "StudyHours": (0, 16),            # hours per day
    "AttendancePercentage": (0, 100),
    "PreviousExamScore": (0, 100),
    "AssignmentsCompleted": (0, 100), # percent completed
    "SleepHours": (2, 14),            # hours per night
    "ExtracurricularHours": (0, 40),  # hours per week
    "ClassParticipation": (0, 10),    # rating scale
    "PreviousBacklogs": (0, 20),      # count of failed courses
    "PostExamConfidence": (1, 10),    # rating scale
    TARGET: (0, 100),
}
