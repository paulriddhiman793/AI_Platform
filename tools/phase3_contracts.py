"""
Phase 3 execution contracts: required keys for pipeline outputs.
"""

DA_REQUIRED_KEYS = [
    "run_timestamp",
    "dataset_path",
    "overview",
    "data_quality",
    "descriptive_statistics",
    "missing_value_analysis",
    "outlier_analysis",
    "distributions",
    "boxplot_statistics",
    "duplicate_analysis",
    "dtype_audit",
    "date_analysis",
    "cardinality_audit",
    "anomaly_flags",
    "string_quality",
    "correlation_analysis",
    "chi_square_tests",
    "categorical_frequencies",
    "time_series_trends",
    "hypothesis_tests",
    "segment_profiles",
    "kpi_summary",
    "cohort_analysis",
]

ML_REQUIRED_KEYS = [
    "run_timestamp",
    "task_type",
    "target_col",
    "n_train",
    "n_test",
    "n_cv_folds",
    "optuna_trials",
    "results",
    "best_model",
]

