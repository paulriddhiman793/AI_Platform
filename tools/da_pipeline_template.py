"""
Data Analyst Pipeline - Automated Exploratory Data Analysis (v2 hardened)
===========================================================================
Fixed gaps vs v1:

  CORRECTNESS
   [F1]  col_type_tag: datetime detection now uses errors="coerce" + threshold
         instead of raising on first bad value - avoids misclassifying columns
         with even one valid-looking date as datetime_string
   [F2]  skew_label: skewness == 0.0 was falsy -> fell through to "approximately
         symmetric" accidentally; now uses explicit None check
   [F3]  Outlier IQR: when IQR == 0 (constant-like column) division would
         produce infinite fences; zero-IQR case now skipped cleanly
   [F4]  Correlation pearsonr alignment was incorrect - .align() on two
         independently-dropna'd series can misalign indices; fixed to use
         pairwise dropna on the joint dataframe
   [F5]  dtype audit: pd.to_numeric called on full head(200) - a single
         non-numeric value in head causes false-negative; now uses errors="coerce"
         and checks conversion rate instead
   [F6]  Z-score on constant column -> division by zero; guarded with std > 0
   [F7]  Cardinality tag logic had dead branch - "low_card" was unreachable
         because high_card caught >20 and binary caught ==2, leaving only
         3..20 which was never reached; fixed tag ordering

  ROBUSTNESS
   [R1]  Empty dataframe guard - all sections now skip gracefully if df is empty
   [R2]  Single-row dataframe: std/skew/kurtosis return NaN; safe_float handles
         but skew_label would crash on None; now guarded
   [R3]  All-NaN numeric column: dropna() -> empty series; every section already
         skips len==0 but IQR section checked len<4, not len==0 - unified to
         len < 4 with explicit log
   [R4]  Hypothesis test: groups[0]/groups[1] after .unique() is unordered;
         added sort for reproducibility
   [R5]  Shapiro-Wilk only valid for n >= 3; added floor guard; also added
         D'Agostino-Pearson test as fallback for n > 5000 where Shapiro is slow
   [R6]  co_missingness: bitwise & on int columns is correct but silently
         wrong if miss_df has float dtype after astype; forced int explicitly
   [R7]  date_analysis: future date detection was a comment placeholder -
         now actually implemented and included in anomaly_flags

  COVERAGE GAPS
   [C1]  Multi-group hypothesis testing (ANOVA + Kruskal-Wallis) added for
         categorical columns with 3-10 groups, not just binary ones
   [C2]  Chi-square test of independence added between all categorical column
         pairs - detects associations between two categoricals
   [C3]  Missing value pattern analysis added - detects if missingness in one
         column correlates with the VALUE of another (MCAR vs MAR signal)
   [C4]  String quality checks added for categorical columns: whitespace padding,
         case inconsistency, empty strings disguised as non-null
   [C5]  Percentile profile added to descriptive stats: p1, p5, p95, p99
         - critical for understanding tail behaviour

  OUTPUT
   [O1]  da_report.txt: human-readable text report now generated alongside JSON
         (same pattern as ML pipeline reports)
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

import scipy.stats as stats

warnings.filterwarnings("ignore")

# ---- Logging ----
LOG_PATH = Path("da_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---- Config ----
DATA_PATH           = r"__DATA_PATH__"
OUTLIER_Z_THRESH    = 3.0
HIGH_CARD_THR       = 20
MAX_CAT_FREQ_SHOW   = 15
HIST_BINS           = 30
SEGMENT_MAX_GROUPS  = 10
CORR_SIG_THRESH     = 0.05
TOP_CORR_PAIRS      = 20
DATETIME_SAMPLE_N   = 200
DATETIME_HIT_RATE   = 0.80

log.info(f"Loading dataset: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
df     = df_raw.copy()
log.info(f"Shape: {df.shape}  |  Columns: {list(df.columns)}")

# ---- Empty dataframe guard [R1] ----
if df.empty:
    log.error("Dataset is empty - nothing to analyse.")
    raise SystemExit("Empty dataset.")

# ---- Helper utilities ----
def safe_float(v):
    """Convert numpy scalars to plain Python float; return None for NaN/Inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def col_type_tag(series: pd.Series) -> str:
    """
    Classify a column. [F1] datetime detection uses hit-rate threshold rather
    than all-or-nothing to avoid misclassifying columns that happen to have
    a few date-like strings among non-date values.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object:
        sample = series.dropna().head(DATETIME_SAMPLE_N)
        if len(sample) > 0:
            parsed    = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            hit_rate  = parsed.notna().mean()
            if hit_rate >= DATETIME_HIT_RATE:
                return "datetime_string"
        types_seen = set(type(v).__name__ for v in sample)
        if len(types_seen) > 1:
            return "mixed"
    return "categorical"

# 1. Dataset overview
log.info("-- 1. Dataset Overview --")
col_tags     = {c: col_type_tag(df[c]) for c in df.columns}
numeric_cols = [c for c, t in col_tags.items() if t == "numeric"]
cat_cols     = [c for c, t in col_tags.items() if t in ("categorical", "boolean")]
dt_cols      = [c for c, t in col_tags.items() if t in ("datetime", "datetime_string")]
mixed_cols   = [c for c, t in col_tags.items() if t == "mixed"]

overview = {
    "n_rows":           int(df.shape[0]),
    "n_cols":           int(df.shape[1]),
    "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 3),
    "numeric_cols":     numeric_cols,
    "categorical_cols": cat_cols,
    "datetime_cols":    dt_cols,
    "mixed_type_cols":  mixed_cols,
    "column_dtypes":    {c: str(df[c].dtype) for c in df.columns},
    "column_type_tags": col_tags,
}
log.info(f"Numeric: {len(numeric_cols)} | Cat: {len(cat_cols)} | "
         f"Datetime: {len(dt_cols)} | Mixed: {len(mixed_cols)}")

# 2. Descriptive statistics [C5]
log.info("-- 2. Descriptive Statistics --")

descriptive = {}
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1, q3   = s.quantile(0.25), s.quantile(0.75)
    skewness = safe_float(s.skew())
    kurt     = safe_float(s.kurtosis())

    # [F2] explicit None check - skewness == 0.0 is falsy
    if skewness is None:
        skew_label = "unknown"
    elif skewness > 1:
        skew_label = "heavy right skew"
    elif skewness > 0.5:
        skew_label = "moderate right skew"
    elif skewness < -1:
        skew_label = "heavy left skew"
    elif skewness < -0.5:
        skew_label = "moderate left skew"
    else:
        skew_label = "approximately symmetric"

    descriptive[col] = {
        "count":    int(s.count()),
        "mean":     safe_float(s.mean()),
        "median":   safe_float(s.median()),
        "std":      safe_float(s.std()),
        "variance": safe_float(s.var()),
        "min":      safe_float(s.min()),
        "max":      safe_float(s.max()),
        "range":    safe_float(s.max() - s.min()),
        "p1":       safe_float(s.quantile(0.01)),
        "p5":       safe_float(s.quantile(0.05)),
        "q1":       safe_float(q1),
        "q3":       safe_float(q3),
        "p95":      safe_float(s.quantile(0.95)),
        "p99":      safe_float(s.quantile(0.99)),
        "iqr":      safe_float(q3 - q1),
        "skewness": skewness,
        "kurtosis": kurt,
        "skew_label": skew_label,
        "cv_pct":   safe_float(s.std() / s.mean() * 100) if s.mean() != 0 else None,
    }

log.info(f"Descriptive stats: {len(descriptive)} numeric columns")

# 3. Missing value analysis [C3]
log.info("-- 3. Missing Value Analysis --")

missing = {}
for col in df.columns:
    n_miss = int(df[col].isnull().sum())
    if n_miss > 0:
        missing[col] = {
            "count":    n_miss,
            "pct":      round(n_miss / len(df) * 100, 2),
            "dtype":    str(df[col].dtype),
            "severity": ("critical" if n_miss / len(df) > 0.5 else
                         "high"     if n_miss / len(df) > 0.2 else
                         "moderate" if n_miss / len(df) > 0.05 else "low"),
        }

# Co-missingness [R6] - forced int dtype
co_missing = {}
miss_cols = list(missing.keys())
if len(miss_cols) >= 2:
    miss_df = df[miss_cols].isnull().astype("int8")
    for i, c1 in enumerate(miss_cols):
        for c2 in miss_cols[i+1:]:
            both = int((miss_df[c1] & miss_df[c2]).sum())
            if both > 0:
                co_missing[f"{c1} & {c2}"] = both

# [C3] MAR signal: does missingness in col A correlate with values in col B?
mar_signals = []
for miss_col in list(missing.keys())[:10]:
    miss_indicator = df[miss_col].isnull().astype(int)
    for num_col in numeric_cols[:10]:
        if num_col == miss_col:
            continue
        paired = pd.concat([miss_indicator, df[num_col]], axis=1).dropna()
        if len(paired) < 10 or paired.iloc[:, 0].std() == 0:
            continue
        try:
            pb_r, pb_p = stats.pointbiserialr(paired.iloc[:, 0], paired.iloc[:, 1])
            if abs(pb_r) >= 0.15 and pb_p < 0.05:
                mar_signals.append({
                    "missing_col":    miss_col,
                    "predictor_col":  num_col,
                    "correlation":    safe_float(pb_r),
                    "p_value":        safe_float(pb_p),
                    "interpretation": "MAR likely - missingness correlates with another variable",
                })
        except Exception:
            pass

missing_summary = {
    "total_missing_cells":  int(df.isnull().sum().sum()),
    "total_missing_pct":    round(df.isnull().sum().sum() / df.size * 100, 2),
    "cols_with_missing":    len(missing),
    "completely_null_cols": [c for c in df.columns if df[c].isnull().all()],
    "per_column":           missing,
    "co_missingness":       co_missing,
    "mar_signals":          mar_signals,
}
log.info(f"Missing: {missing_summary['total_missing_pct']}% | "
         f"{missing_summary['cols_with_missing']} cols | "
         f"{len(mar_signals)} MAR signals")

# 4. Outlier detection [F3] [F6]
log.info("-- 4. Outlier Detection --")

outliers = {}
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) < 4:
        continue

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1

    # [F3] zero-IQR means near-constant column - skip IQR fences
    if iqr == 0:
        log.info(f"  Skipping IQR outliers for '{col}' - IQR=0 (constant-like)")
        continue

    lo, hi       = q1 - 1.5 * iqr,  q3 + 1.5 * iqr
    lo3, hi3     = q1 - 3.0 * iqr,  q3 + 3.0 * iqr
    n_iqr        = int(((s < lo) | (s > hi)).sum())
    n_extreme    = int(((s < lo3) | (s > hi3)).sum())

    # [F6] Z-score only when std > 0
    if s.std() > 0:
        z_scores = np.abs(stats.zscore(s))
        n_zscore = int((z_scores > OUTLIER_Z_THRESH).sum())
    else:
        n_zscore = 0

    if n_iqr > 0:
        outliers[col] = {
            "iqr_outliers":     n_iqr,
            "iqr_outlier_pct":  round(n_iqr / len(s) * 100, 2),
            "zscore_outliers":  n_zscore,
            "extreme_outliers": n_extreme,
            "lower_fence":      safe_float(lo),
            "upper_fence":      safe_float(hi),
            "min_value":        safe_float(s.min()),
            "max_value":        safe_float(s.max()),
            "severity":         ("high"     if n_iqr / len(s) > 0.10 else
                                 "moderate" if n_iqr / len(s) > 0.03 else "low"),
        }

log.info(f"Outlier analysis: {len(outliers)} columns have outliers")

# 5. Duplicate detection
log.info("-- 5. Duplicate Detection --")

n_dup_rows    = int(df.duplicated().sum())
n_dup_full    = int(df.duplicated(keep=False).sum())
n_dup_numeric = int(df[numeric_cols].duplicated().sum()) if numeric_cols else 0

duplicates = {
    "duplicate_rows":        n_dup_rows,
    "duplicate_rows_pct":    round(n_dup_rows / len(df) * 100, 2),
    "rows_involved_in_dups": n_dup_full,
    "numeric_only_dups":     n_dup_numeric,
}
log.info(f"Duplicates: {n_dup_rows} rows ({duplicates['duplicate_rows_pct']}%)")

# 6. Data type audit [F5]
log.info("-- 6. Data Type Audit --")

dtype_issues = []
for col in df.columns:
    s     = df[col]
    issue = None
    # [F5] numeric-as-object: use errors="coerce" and check conversion rate
    if s.dtype == object:
        converted   = pd.to_numeric(s.dropna(), errors="coerce")
        conv_rate   = converted.notna().mean()
        if conv_rate >= 0.95:
            issue = "numeric_as_object"
    # Boolean stored as int
    if s.dtype in [np.int64, np.int32, np.int16, np.int8] and set(s.dropna().unique()).issubset({0, 1}):
        issue = "possible_boolean_as_int"
    # ID-like column
    if s.nunique() == len(df) and s.dtype != object:
        issue = "possible_id_column"
    if issue:
        dtype_issues.append({
            "column":       col,
            "current_dtype":str(s.dtype),
            "issue":        issue,
        })

log.info(f"Dtype issues: {len(dtype_issues)}")

# 7. Date column detection and parsing [R7]
log.info("-- 7. Date Column Analysis --")

date_analysis    = {}
future_date_cols = []
today            = pd.Timestamp.now().normalize()

for col in dt_cols:
    try:
        parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
        valid  = parsed.dropna()
        if len(valid) == 0:
            continue
        n_future = int((valid > today).sum())
        if n_future > 0:
            future_date_cols.append({"column": col, "n_future_dates": n_future})
        date_analysis[col] = {
            "min_date":        str(valid.min().date()),
            "max_date":        str(valid.max().date()),
            "span_days":       int((valid.max() - valid.min()).days),
            "n_parsed":        int(len(valid)),
            "n_failed":        int(len(parsed) - len(valid)),
            "n_future_dates":  n_future,
            "unique_months":   int(valid.dt.to_period("M").nunique()),
            "unique_years":    int(valid.dt.year.nunique()),
            "has_future_dates":n_future > 0,
        }
    except Exception as e:
        log.warning(f"Date parsing failed for '{col}': {e}")

log.info(f"Date columns analysed: {len(date_analysis)}")

# 8. Cardinality audit [F7]
log.info("-- 8. Cardinality Audit --")

cardinality = {}
for col in df.columns:
    n_unique = int(df[col].nunique())
    n_rows   = len(df)
    # [F7] tag ordering fixed: constant -> binary -> id_like -> low_card -> high_card
    if n_unique == 0 or n_unique == 1:
        tag = "constant"
    elif n_unique == 2:
        tag = "binary"
    elif n_unique == n_rows:
        tag = "id_like"
    elif n_unique <= HIGH_CARD_THR:
        tag = "low_card"
    else:
        tag = "high_card"

    cardinality[col] = {
        "n_unique":   n_unique,
        "pct_unique": round(n_unique / n_rows * 100, 2),
        "tag":        tag,
    }

constant_cols = [c for c, v in cardinality.items() if v["tag"] == "constant"]
id_cols       = [c for c, v in cardinality.items() if v["tag"] == "id_like"]
log.info(f"Constant: {constant_cols} | ID-like: {id_cols}")

# 9. Anomaly flagging [R7] [C4]
log.info("-- 9. Anomaly Flagging --")

anomalies = []

# Numeric anomalies
neg_keywords  = ["price", "age", "count", "qty", "quantity", "amount",
                 "salary", "income", "area", "size", "sqft", "population",
                 "weight", "height", "distance", "duration", "rate"]
zero_keywords = ["price", "salary", "income", "area", "sqft", "revenue",
                 "cost", "value"]

for col in numeric_cols:
    s = df[col].dropna()
    if any(k in col.lower() for k in neg_keywords) and (s < 0).any():
        anomalies.append({
            "column":  col, "type": "negative_where_unexpected",
            "count":   int((s < 0).sum()),
            "example": safe_float(s[s < 0].iloc[0]),
        })
    if any(k in col.lower() for k in zero_keywords) and (s == 0).any():
        anomalies.append({
            "column": col, "type": "zero_where_suspicious",
            "count":  int((s == 0).sum()),
        })

# Future date anomalies [R7]
for fd in future_date_cols:
    anomalies.append({
        "column": fd["column"],
        "type":   "future_dates",
        "count":  fd["n_future_dates"],
    })

# Mixed type anomalies
for col in mixed_cols:
    anomalies.append({
        "column": col, "type": "mixed_data_types",
        "detail": "Column contains values of multiple Python types",
    })

# [C4] String quality checks for categorical columns
string_quality = {}
for col in cat_cols:
    s    = df[col].dropna().astype(str)
    if len(s) == 0:
        continue
    issues = []
    # Empty strings masquerading as non-null
    n_empty = int((s.str.strip() == "").sum())
    if n_empty > 0:
        issues.append({"issue": "empty_strings", "count": n_empty})
    # Whitespace padding
    n_padded = int((s != s.str.strip()).sum())
    if n_padded > 0:
        issues.append({"issue": "whitespace_padding", "count": n_padded})
    # Case inconsistency (same value in different cases)
    lower_uniq  = s.str.lower().nunique()
    actual_uniq = s.nunique()
    if lower_uniq < actual_uniq:
        issues.append({
            "issue":           "case_inconsistency",
            "unique_as_typed": actual_uniq,
            "unique_lowercased": lower_uniq,
            "extra_variants":  actual_uniq - lower_uniq,
        })
    if issues:
        string_quality[col] = issues
        anomalies.append({
            "column": col, "type": "string_quality_issues",
            "detail": issues,
        })

log.info(f"Anomalies flagged: {len(anomalies)} | String quality issues: {len(string_quality)} cols")

# 10. Distribution summaries [R5]
log.info("-- 10. Distribution Summaries --")

distributions = {}
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) < 2:
        continue
    counts, edges = np.histogram(s, bins=HIST_BINS)

    is_normal, p_normal, norm_test_used = None, None, None

    if len(s) < 3:
        norm_test_used = "skipped_too_few"
    elif len(s) <= 5000:
        # [R5] Shapiro-Wilk - valid range 3..5000
        try:
            _, p_sw      = stats.shapiro(s)
            is_normal    = bool(p_sw > 0.05)
            p_normal     = safe_float(p_sw)
            norm_test_used = "shapiro_wilk"
        except Exception:
            pass
    else:
        # [R5] D'Agostino-Pearson for large samples
        try:
            _, p_dag     = stats.normaltest(s)
            is_normal    = bool(p_dag > 0.05)
            p_normal     = safe_float(p_dag)
            norm_test_used = "dagostino_pearson"
        except Exception:
            pass

    distributions[col] = {
        "histogram_counts":  counts.tolist(),
        "histogram_edges":   [safe_float(e) for e in edges],
        "is_normal":         is_normal,
        "normality_p":       p_normal,
        "normality_test":    norm_test_used,
        "normality_verdict": ("normal" if is_normal else "non-normal")
                             if is_normal is not None else "untested",
    }

log.info(f"Distributions: {len(distributions)} columns")

# 11. Box plot statistics
log.info("-- 11. Box Plot Statistics --")

boxplots = {}
for col in numeric_cols:
    s = df[col].dropna()
    if len(s) < 4:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_pts = s[(s < lo) | (s > hi)]
    boxplots[col] = {
        "min":           safe_float(s.min()),
        "q1":            safe_float(q1),
        "median":        safe_float(s.median()),
        "q3":            safe_float(q3),
        "max":           safe_float(s.max()),
        "iqr":           safe_float(iqr),
        "lower_whisker": safe_float(s[s >= lo].min()),
        "upper_whisker": safe_float(s[s <= hi].max()),
        "n_outliers":    int(len(outlier_pts)),
        "outlier_sample":[safe_float(v) for v in outlier_pts.head(10).tolist()],
    }

# 12. Correlation matrix [F4]
log.info("-- 12. Correlation Analysis --")

corr_data = {}
if len(numeric_cols) >= 2:
    num_df   = df[numeric_cols].copy()
    pearson  = num_df.corr(method="pearson")
    spearman = num_df.corr(method="spearman")

    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            p_r = safe_float(pearson.loc[c1, c2])
            s_r = safe_float(spearman.loc[c1, c2])
            # [F4] correct pairwise dropna - drop rows where EITHER col is NaN
            pair_df = num_df[[c1, c2]].dropna()
            p_val   = None
            if len(pair_df) >= 3 and p_r is not None:
                try:
                    _, p_val = stats.pearsonr(pair_df[c1], pair_df[c2])
                    p_val    = safe_float(p_val)
                except Exception:
                    pass
            if p_r is not None:
                pairs.append({
                    "col1":        c1,
                    "col2":        c2,
                    "pearson_r":   p_r,
                    "spearman_r":  s_r,
                    "abs_r":       round(abs(p_r), 6),
                    "p_value":     p_val,
                    "n_pairs":     int(len(pair_df)),
                    "significant": bool(p_val < CORR_SIG_THRESH) if p_val is not None else None,
                    "strength":    ("very strong" if abs(p_r) >= 0.8 else
                                    "strong"      if abs(p_r) >= 0.6 else
                                    "moderate"    if abs(p_r) >= 0.4 else
                                    "weak"        if abs(p_r) >= 0.2 else "negligible"),
                    "direction":   "positive" if p_r >= 0 else "negative",
                })

    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    corr_data = {
        "pearson_matrix":   {c: {c2: safe_float(pearson.loc[c, c2])
                                 for c2 in numeric_cols} for c in numeric_cols},
        "spearman_matrix":  {c: {c2: safe_float(spearman.loc[c, c2])
                                 for c2 in numeric_cols} for c in numeric_cols},
        "top_pairs":        pairs[:TOP_CORR_PAIRS],
        "strong_pairs":     [p for p in pairs if p["abs_r"] >= 0.6],
        "significant_pairs":[p for p in pairs if p.get("significant")],
    }
    log.info(f"Correlation: {len(pairs)} pairs | "
             f"{len(corr_data['strong_pairs'])} strong | "
             f"{len(corr_data['significant_pairs'])} significant")

# 12b. Chi-square test between categorical pairs [C2]
log.info("-- 12b. Categorical Association (Chi-Square) --")

chi_square_tests = []
cat_cols_filtered = [c for c in cat_cols if 2 <= df[c].nunique() <= HIGH_CARD_THR]

for i, c1 in enumerate(cat_cols_filtered):
    for c2 in cat_cols_filtered[i+1:]:
        try:
            ct       = pd.crosstab(df[c1], df[c2])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            # Cramer's V effect size
            n        = ct.sum().sum()
            min_dim  = min(ct.shape) - 1
            cramers_v = safe_float(np.sqrt(chi2 / (n * min_dim))) if (n * min_dim) > 0 else None
            chi_square_tests.append({
                "col1":       c1,
                "col2":       c2,
                "chi2":       safe_float(chi2),
                "p_value":    safe_float(p),
                "dof":        int(dof),
                "cramers_v":  cramers_v,
                "significant":bool(p < CORR_SIG_THRESH),
                "association":("strong"   if cramers_v and cramers_v >= 0.5 else
                               "moderate" if cramers_v and cramers_v >= 0.3 else
                               "weak"     if cramers_v and cramers_v >= 0.1 else
                               "negligible") if cramers_v else "unknown",
            })
        except Exception as e:
            log.warning(f"Chi-square failed [{c1} x {c2}]: {e}")

chi_square_tests.sort(key=lambda x: x.get("p_value") or 1.0)
log.info(f"Chi-square: {len(chi_square_tests)} pairs tested | "
         f"{sum(1 for t in chi_square_tests if t['significant'])} significant")

# 13. Categorical frequency analysis
log.info("-- 13. Categorical Frequency Analysis --")

cat_frequencies = {}
for col in cat_cols:
    vc    = df[col].value_counts(dropna=False)
    total = len(df)
    cat_frequencies[col] = {
        "n_unique":       int(df[col].nunique()),
        "top_categories": [
            {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
            for k, v in vc.head(MAX_CAT_FREQ_SHOW).items()
        ],
        "mode":           str(vc.index[0]) if len(vc) > 0 else None,
        "mode_pct":       round(vc.iloc[0] / total * 100, 2) if len(vc) > 0 else None,
        "entropy":        safe_float(stats.entropy(vc.values)),
    }

log.info(f"Categorical frequency: {len(cat_frequencies)} columns")

# 14. Time series trends
log.info("-- 14. Time Series Trends --")

time_trends = {}
for dt_col in dt_cols:
    try:
        parsed = pd.to_datetime(df[dt_col], infer_datetime_format=True, errors="coerce")
        temp   = df.copy()
        temp["_dt"] = parsed
        temp   = temp.dropna(subset=["_dt"])
        if len(temp) == 0:
            continue
        temp["_month"] = temp["_dt"].dt.to_period("M").astype(str)
        monthly = {}
        for num_col in numeric_cols[:5]:
            grp = temp.groupby("_month")[num_col].agg(["mean", "sum", "count"])
            monthly[num_col] = grp.reset_index().rename(
                columns={"_month": "month", "mean": "avg", "sum": "total", "count": "n"}
            ).to_dict(orient="records")
        time_trends[dt_col] = {
            "monthly_aggregates": monthly,
            "n_periods":          int(temp["_month"].nunique()),
        }
    except Exception as e:
        log.warning(f"Time series failed for '{dt_col}': {e}")

log.info(f"Time series: {len(time_trends)} date columns")

# 15. Hypothesis testing - binary AND multi-group [C1] [R4]
log.info("-- 15. Hypothesis Testing --")

hypothesis_tests = {}

# Binary groups: t-test + Mann-Whitney
binary_cats = [c for c in cat_cols if df[c].nunique() == 2]
for cat_col in binary_cats[:5]:
    groups   = sorted(df[cat_col].dropna().unique().tolist())
    g1_label = str(groups[0])
    g2_label = str(groups[1])
    tests    = []
    for num_col in numeric_cols:
        g1 = df[df[cat_col] == groups[0]][num_col].dropna()
        g2 = df[df[cat_col] == groups[1]][num_col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        try:
            _, p_lev    = stats.levene(g1, g2)
            equal_var   = bool(p_lev > 0.05)
            t_stat, p_t = stats.ttest_ind(g1, g2, equal_var=equal_var)
            u_stat, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            pooled_std  = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            cohens_d    = safe_float((g1.mean() - g2.mean()) / pooled_std) \
                          if pooled_std > 0 else None
            tests.append({
                "numeric_col":        num_col,
                "group1_label":       g1_label,
                "group2_label":       g2_label,
                "group1_mean":        safe_float(g1.mean()),
                "group2_mean":        safe_float(g2.mean()),
                "group1_median":      safe_float(g1.median()),
                "group2_median":      safe_float(g2.median()),
                "t_statistic":        safe_float(t_stat),
                "t_pvalue":           safe_float(p_t),
                "mannwhitney_u":      safe_float(u_stat),
                "mannwhitney_p":      safe_float(p_u),
                "cohens_d":           cohens_d,
                "effect_size":        ("large"       if cohens_d and abs(cohens_d) >= 0.8 else
                                       "medium"      if cohens_d and abs(cohens_d) >= 0.5 else
                                       "small"       if cohens_d and abs(cohens_d) >= 0.2 else
                                       "negligible") if cohens_d else "unknown",
                "significant_ttest":  bool(p_t < CORR_SIG_THRESH) if p_t is not None else None,
                "significant_mw":     bool(p_u < CORR_SIG_THRESH) if p_u is not None else None,
            })
        except Exception as e:
            log.warning(f"t-test failed [{cat_col} x {num_col}]: {e}")

    if tests:
        hypothesis_tests[cat_col] = {
            "test_type":       "binary",
            "group1":          g1_label,
            "group2":          g2_label,
            "n_group1":        int((df[cat_col] == groups[0]).sum()),
            "n_group2":        int((df[cat_col] == groups[1]).sum()),
            "tests":           sorted(tests, key=lambda x: x.get("t_pvalue") or 1.0),
            "significant_cols":[t["numeric_col"] for t in tests if t.get("significant_ttest")],
        }

# [C1] Multi-group: ANOVA + Kruskal-Wallis for 3-10 group categoricals
multi_cats = [c for c in cat_cols if 3 <= df[c].nunique() <= SEGMENT_MAX_GROUPS]
for cat_col in multi_cats[:5]:
    anova_tests = []
    for num_col in numeric_cols:
        groups_data = [
            grp[num_col].dropna().values
            for _, grp in df.groupby(cat_col)
            if len(grp[num_col].dropna()) >= 3
        ]
        if len(groups_data) < 2:
            continue
        try:
            f_stat, p_anova = stats.f_oneway(*groups_data)
            h_stat, p_kw    = stats.kruskal(*groups_data)
            # Eta-squared effect size for ANOVA
            grand_mean      = df[num_col].dropna().mean()
            ss_between      = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_data)
            ss_total        = sum(((v - grand_mean)**2) for g in groups_data for v in g)
            eta_sq          = safe_float(ss_between / ss_total) if ss_total > 0 else None
            anova_tests.append({
                "numeric_col":       num_col,
                "f_statistic":       safe_float(f_stat),
                "anova_p":           safe_float(p_anova),
                "kruskal_h":         safe_float(h_stat),
                "kruskal_p":         safe_float(p_kw),
                "eta_squared":       eta_sq,
                "effect_size":       ("large"  if eta_sq and eta_sq >= 0.14 else
                                      "medium" if eta_sq and eta_sq >= 0.06 else
                                      "small"  if eta_sq and eta_sq >= 0.01 else
                                      "negligible") if eta_sq else "unknown",
                "significant_anova": bool(p_anova < CORR_SIG_THRESH) if p_anova is not None else None,
                "significant_kw":    bool(p_kw    < CORR_SIG_THRESH) if p_kw    is not None else None,
            })
        except Exception as e:
            log.warning(f"ANOVA failed [{cat_col} x {num_col}]: {e}")

    if anova_tests:
        hypothesis_tests[f"{cat_col}_multigroup"] = {
            "test_type":           "multi_group",
            "categorical_col":     cat_col,
            "n_groups":            int(df[cat_col].nunique()),
            "group_labels":        [str(v) for v in sorted(df[cat_col].dropna().unique().tolist())],
            "tests":               sorted(anova_tests, key=lambda x: x.get("anova_p") or 1.0),
            "significant_cols_anova": [t["numeric_col"] for t in anova_tests
                                        if t.get("significant_anova")],
        }

log.info(f"Hypothesis tests: {len(hypothesis_tests)} groups tested")

# 16. Segment profiling
log.info("-- 16. Segment Profiling --")

segment_profiles = {}
for cat_col in cat_cols:
    n_groups = df[cat_col].nunique()
    if n_groups < 2 or n_groups > SEGMENT_MAX_GROUPS:
        continue
    profiles = {}
    for grp_val, grp_df in df.groupby(cat_col):
        grp_stats = {"n": int(len(grp_df)), "pct": round(len(grp_df)/len(df)*100, 2)}
        for num_col in numeric_cols[:8]:
            s = grp_df[num_col].dropna()
            if len(s) == 0:
                continue
            grp_stats[num_col] = {
                "mean":   safe_float(s.mean()),
                "median": safe_float(s.median()),
                "std":    safe_float(s.std()),
                "min":    safe_float(s.min()),
                "max":    safe_float(s.max()),
            }
        profiles[str(grp_val)] = grp_stats
    segment_profiles[cat_col] = profiles

log.info(f"Segment profiles: {len(segment_profiles)} categorical columns")

# 17. KPI summary
log.info("-- 17. KPI Summary --")

kpi_keywords = ["price", "revenue", "sales", "amount", "income", "salary",
                "cost", "profit", "value", "score", "rating", "age",
                "count", "qty", "quantity", "area", "sqft", "size"]
kpi_summary  = {}
for col in numeric_cols:
    s = df[col].dropna()
    kpi_summary[col] = {
        "is_likely_kpi": any(k in col.lower() for k in kpi_keywords),
        "total":         safe_float(s.sum()),
        "mean":          safe_float(s.mean()),
        "median":        safe_float(s.median()),
        "count":         int(s.count()),
        "pct_nonzero":   round((s != 0).sum() / len(s) * 100, 2) if len(s) > 0 else None,
    }

log.info(f"KPI summary: {len(kpi_summary)} numeric columns")

# 18. Cohort analysis
log.info("-- 18. Cohort Analysis --")

cohort_analysis = {}
if dt_cols and numeric_cols:
    dt_col = dt_cols[0]
    try:
        parsed  = pd.to_datetime(df[dt_col], infer_datetime_format=True, errors="coerce")
        temp    = df.copy()
        temp["_period"] = parsed.dt.to_period("M").astype(str)
        temp    = temp.dropna(subset=["_period"])
        if len(temp) > 0:
            kpi_cols = [c for c in numeric_cols
                        if any(k in c.lower() for k in kpi_keywords)][:3] or numeric_cols[:3]
            cohort_rows = []
            for period, grp in temp.groupby("_period"):
                row = {"period": str(period), "n": int(len(grp))}
                for kc in kpi_cols:
                    s = grp[kc].dropna()
                    row[f"{kc}_mean"]  = safe_float(s.mean())
                    row[f"{kc}_total"] = safe_float(s.sum())
                cohort_rows.append(row)
            cohort_analysis = {
                "date_column":     dt_col,
                "kpi_columns":     kpi_cols,
                "monthly_cohorts": sorted(cohort_rows, key=lambda x: x["period"]),
                "n_periods":       len(cohort_rows),
            }
            log.info(f"Cohort: {len(cohort_rows)} periods")
    except Exception as e:
        log.warning(f"Cohort analysis failed: {e}")
else:
    log.info("Cohort skipped - no date columns")

# Data quality score

def compute_quality_score() -> float:
    score = 100.0
    score -= min(30, missing_summary["total_missing_pct"] * 2)
    score -= min(15, duplicates["duplicate_rows_pct"] * 3)
    score -= len(constant_cols) * 3
    score -= min(20, sum(1 for v in outliers.values() if v["severity"] == "high") * 4)
    score -= min(10, len(dtype_issues) * 2)
    score -= min(10, len(anomalies) * 2)
    # [C4] string quality issues also penalise
    score -= min(5, len(string_quality) * 1)
    return round(max(0.0, score), 1)

quality_score = compute_quality_score()
quality_label = ("Excellent" if quality_score >= 85 else
                 "Good"      if quality_score >= 70 else
                 "Fair"      if quality_score >= 50 else "Poor")
log.info(f"Data Quality Score: {quality_score}/100 ({quality_label})")

# [O1] Human-readable text report

def build_text_report() -> str:
    W = 72
    lines = [
        "=" * W,
        "  DATA ANALYST PIPELINE - AUTOMATED EDA REPORT  (v2)",
        "=" * W,
        f"  Run timestamp   : {datetime.utcnow().isoformat()}Z",
        f"  Dataset          : {DATA_PATH}",
        f"  Shape            : {overview['n_rows']} rows x {overview['n_cols']} cols",
        f"  Memory           : {overview['memory_mb']} MB",
        "=" * W, "",
        "  DATA QUALITY SCORE",
        "  " + "-" * 44,
        f"  {quality_score}/100  ({quality_label})",
        f"  Missing cells    : {missing_summary['total_missing_pct']}%",
        f"  Duplicate rows   : {duplicates['duplicate_rows']} ({duplicates['duplicate_rows_pct']}%)",
        f"  Dtype issues     : {len(dtype_issues)}",
        f"  Anomalies found  : {len(anomalies)}",
        f"  Constant cols    : {constant_cols or 'None'}",
        "",
        "  COLUMN OVERVIEW",
        "  " + "-" * 44,
        f"  Numeric     : {numeric_cols}",
        f"  Categorical : {cat_cols}",
        f"  Datetime    : {dt_cols}",
        f"  Mixed type  : {mixed_cols or 'None'}",
        "",
    ]

    # Missing value table
    if missing:
        lines += ["  MISSING VALUES", "  " + "-" * 44,
                  f"  {'Column':<30} {'Count':>7} {'%':>7} {'Severity':<10}"]
        for col, m in sorted(missing.items(), key=lambda x: -x[1]["pct"]):
            lines.append(f"  {col:<30} {m['count']:>7} {m['pct']:>6.1f}% {m['severity']:<10}")
        lines.append("")

    # Top anomalies
    if anomalies:
        lines += ["  ANOMALIES", "  " + "-" * 44]
        for a in anomalies[:15]:
            lines.append(f"  ! [{a['type']}]  {a['column']}"
                         + (f"  - {a.get('count','')} occurrences" if "count" in a else ""))
        lines.append("")

    # Descriptive stats table
    lines += ["  DESCRIPTIVE STATISTICS (numeric columns)", "  " + "-" * 44,
              f"  {'Column':<28} {'Mean':>12} {'Median':>12} {'Std':>12} {'Skew':<22}"]
    for col, d in descriptive.items():
        lines.append(
            f"  {col:<28} {str(d['mean']):>12} {str(d['median']):>12} "
            f"{str(d['std']):>12} {d['skew_label']:<22}"
        )
    lines.append("")

    # Top correlations
    if corr_data.get("top_pairs"):
        lines += ["  TOP CORRELATIONS", "  " + "-" * 44,
                  f"  {'Col 1':<22} {'Col 2':<22} {'Pearson':>8} {'Strength':<14} {'Sig':>5}"]
        for p in corr_data["top_pairs"][:10]:
            sig = "Y" if p.get("significant") else " "
            lines.append(
                f"  {p['col1']:<22} {p['col2']:<22} "
                f"{p['pearson_r']:>8.4f} {p['strength']:<14} {sig:>5}"
            )
        lines.append("")

    # Chi-square top associations
    if chi_square_tests:
        sig_chi = [t for t in chi_square_tests if t["significant"]]
        if sig_chi:
            lines += ["  SIGNIFICANT CATEGORICAL ASSOCIATIONS (Chi-Square)", "  " + "-" * 44,
                      f"  {'Col 1':<22} {'Col 2':<22} {'Cramer V':>9} {'Assoc':<12}"]
            for t in sig_chi[:8]:
                lines.append(
                    f"  {t['col1']:<22} {t['col2']:<22} "
                    f"{str(t['cramers_v']):>9} {t['association']:<12}"
                )
            lines.append("")

    # Hypothesis test summary
    if hypothesis_tests:
        lines += ["  HYPOTHESIS TEST SUMMARY", "  " + "-" * 44]
        for grp_key, ht in list(hypothesis_tests.items())[:5]:
            sig_cols = ht.get("significant_cols") or ht.get("significant_cols_anova", [])
            lines.append(f"  {grp_key}  [{ht['test_type']}]")
            if sig_cols:
                lines.append(f"    Significant differences in: {sig_cols}")
            else:
                lines.append("    No significant differences found")
        lines.append("")

    lines += ["", "=" * W]
    return "\n".join(lines)

report_text = build_text_report()
report_path = Path("da_report.txt")
report_path.write_text(report_text, encoding="utf-8")
log.info(f"Text report saved to {report_path.resolve()}")

# Assemble final payload
payload = {
    "run_timestamp":          datetime.utcnow().isoformat() + "Z",
    "dataset_path":           str(DATA_PATH),
    "overview":               overview,
    "data_quality": {
        "score":   quality_score,
        "label":   quality_label,
        "issues_summary": {
            "missing_pct":       missing_summary["total_missing_pct"],
            "duplicate_rows":    duplicates["duplicate_rows"],
            "dtype_issues":      len(dtype_issues),
            "anomalies":         len(anomalies),
            "string_quality_cols":list(string_quality.keys()),
            "constant_cols":     constant_cols,
            "id_like_cols":      id_cols,
            "mar_signals":       len(mar_signals),
        },
    },
    # Understanding and exploration
    "descriptive_statistics":  descriptive,
    "missing_value_analysis":  missing_summary,
    "outlier_analysis":        outliers,
    "distributions":           distributions,
    "boxplot_statistics":      boxplots,
    # Cleaning and preparation
    "duplicate_analysis":      duplicates,
    "dtype_audit":             dtype_issues,
    "date_analysis":           date_analysis,
    "cardinality_audit":       cardinality,
    "anomaly_flags":           anomalies,
    "string_quality":          string_quality,
    # Visualization data
    "correlation_analysis":    corr_data,
    "chi_square_tests":        chi_square_tests,
    "categorical_frequencies": cat_frequencies,
    "time_series_trends":      time_trends,
    # Business insights
    "hypothesis_tests":        hypothesis_tests,
    "segment_profiles":        segment_profiles,
    "kpi_summary":             kpi_summary,
    "cohort_analysis":         cohort_analysis,
}

# Validate required payload keys
_required_keys = [
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
_missing = [k for k in _required_keys if k not in payload]
if _missing:
    log.warning(f"Payload missing keys: {_missing}")

print("__DA_RESULTS_BEGIN__")
print(json.dumps(payload, indent=2, default=str))
print("__DA_RESULTS_END__")

out_path = Path("da_results.json")
out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
log.info(f"JSON saved to {out_path.resolve()}")
log.info("Data Analysis Pipeline complete.")
