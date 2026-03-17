from model_transparency import run_pipeline
from xgboost import XGBClassifier
import numpy as np

# Small dataset: predict if a student passes (1) or fails (0)
# Features: [hours_studied, attendance_%, prev_score, assignments_done]
X = np.array([
    [1.0, 40, 35, 2],
    [2.0, 55, 50, 3],
    [1.5, 45, 40, 2],
    [3.0, 70, 60, 5],
    [4.0, 80, 72, 6],
    [5.0, 85, 78, 7],
    [6.0, 90, 85, 8],
    [7.0, 92, 88, 9],
    [5.5, 88, 80, 8],
    [2.5, 60, 55, 4],
    [3.5, 75, 65, 6],
    [1.0, 38, 30, 1],
    [4.5, 82, 74, 7],
    [6.5, 91, 87, 9],
    [2.0, 50, 45, 3],
    [3.0, 68, 58, 5],
    [7.0, 95, 92, 10],
    [1.5, 42, 38, 2],
    [5.0, 84, 76, 7],
    [4.0, 78, 68, 6],
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1])

feature_names = ["hours_studied", "attendance_pct", "prev_score", "assignments_done"]

model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

run_pipeline(
    model,
    X, y,
    feature_names=feature_names,
    task_type="classification",
    test_size=0.25,
    scale=False,       # XGBoost handles raw features fine
    cv=3,              # small cv since dataset is small
    n_walkthrough=3
)