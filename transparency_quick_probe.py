"""
Fast model transparency probe used by agents before code generation.
Runs a small sklearn pipeline through model_transparency.run_pipeline.
"""
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from model_transparency import run_pipeline


def main() -> None:
    # Windows-safe console encoding for rich unicode output from model_transparency.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = load_breast_cancer()
    # Keep sample small for speed while preserving signal.
    X = data.data[:180]
    y = data.target[:180]
    feature_names = list(data.feature_names)

    model = LogisticRegression(max_iter=500, random_state=42)
    run_pipeline(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        task_type="classification",
        test_size=0.2,
        scale=True,
        cv=2,
        n_walkthrough=1,
    )


if __name__ == "__main__":
    main()
