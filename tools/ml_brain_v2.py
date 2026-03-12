#!/usr/bin/env python3
"""
ML ENGINEER BRAIN v2 - All Scenarios Covered, Zero API Calls

Usage:
    python tools/ml_brain_v2.py --report pipeline_output.txt --csv data.csv
    python tools/ml_brain_v2.py --report pipeline_output.txt --csv data.csv --apply
    cat output.txt | python tools/ml_brain_v2.py --stdin --csv data.csv
"""

import argparse
import re
import sys
import textwrap
import time
from pathlib import Path


# =============================================================================
#  SECTION 1 - PARSER
# =============================================================================

def _float(text, *pats):
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except Exception:
                pass
    return None


def _int(text, *pats):
    v = _float(text, *pats)
    return int(v) if v is not None else None


def _str(text, *pats):
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _bool_present(text, positive_pat, negative_pat=None):
    found = bool(re.search(positive_pat, text, re.IGNORECASE))
    if not found:
        return False
    if negative_pat and re.search(negative_pat, text, re.IGNORECASE):
        return False
    return True


def parse_report(text: str) -> dict:
    r = {}

    # Task type
    raw_task = _str(text, r"task[_\s]*type\s*[:\-]\s*(\w+)", r"TASK\s*[:\-]\s*(\w+)") or ""
    if "classif" in raw_task.lower():
        r["task"] = "classification"
    elif "regress" in raw_task.lower():
        r["task"] = "regression"
    elif re.search(r"accuracy|f1|auc|roc|precision|recall|logloss|log.loss", text, re.I):
        r["task"] = "classification"
    else:
        r["task"] = "regression"

    # Model
    r["model"] = _str(
        text,
        r"Model\s*[:\-]\s*([\w]+(?:Regressor|Classifier|Boost|Forest|Tree|SVM|Linear|Ridge|Lasso|KNN|MLP|Net)[\w]*)",
        r"Training\s+([\w]+(?:Regressor|Classifier|Boost|Forest|SVM|MLP)[\w]*)",
        r"Model\s*[:\-]\s*(CatBoost\w*|XGBoost\w*|LightGBM\w*|LGBM\w*|RandomForest\w*|"
        r"LinearRegression|LogisticRegression|SVM|SVR|KNN|KNeighbors\w*|MLP\w*|"
        r"Ridge|Lasso|ElasticNet|GradientBoosting\w*|AdaBoost\w*|ExtraTrees\w*|"
        r"HistGradientBoosting\w*|DecisionTree\w*)",
    ) or "Unknown"

    # Dataset
    r["n_samples"] = _int(text, r"Total samples\s*[:\-]\s*([\d,]+)", r"Samples\s*[:\-]\s*([\d,]+)", r"\((\d+),\s*\d+\)")
    r["n_features"] = _int(text, r"Features\s*[:\-]\s*(\d+)")
    r["n_train"] = _int(text, r"Train(?:ing)? samples\s*[:\-]\s*([\d,]+)", r"Train\s*[:\-]\s*(\d+)")
    r["n_test"] = _int(text, r"Test\s+samples\s*[:\-]\s*([\d,]+)", r"Test\s*[:\-]\s*(\d+)")
    r["test_ratio"] = _float(text, r"Test\s+ratio\s*[:\-]\s*([\d.]+)%?", r"test.*?(\d+)%")
    r["gpu_hint"] = bool(re.search(r"gpu|cuda|device.*gpu", text, re.I))

    # Regression metrics
    r["train_r2"] = _float(text, r"Train R\s*2\s*[:\-]\s*(-?[\d.]+)", r"Train R\^2\s*[:\-]\s*(-?[\d.]+)")
    r["test_r2"] = _float(text, r"Test\s+R\s*2\s*[:\-]\s*(-?[\d.]+)", r"Test\s+R\^2\s*[:\-]\s*(-?[\d.]+)")
    r["r2_gap"] = _float(text, r"R\s*2\s*gap\s*[:\-]\s*[+]?([-\d.]+)", r"R\^2\s*gap\s*[:\-]\s*[+]?([-\d.]+)")
    r["train_rmse"] = _float(text, r"Train RMSE\s*[:\-]\s*([\d.]+)")
    r["test_rmse"] = _float(text, r"Test\s+RMSE\s*[:\-]\s*([\d.]+)")
    r["train_mae"] = _float(text, r"Train MAE\s*[:\-]\s*([\d.]+)")
    r["test_mae"] = _float(text, r"Test\s+MAE\s*[:\-]\s*([\d.]+)")
    r["train_mape"] = _float(text, r"Train MAPE\s*[:\-]\s*([\d.]+)")
    r["test_mape"] = _float(text, r"Test\s+MAPE\s*[:\-]\s*([\d.]+)")
    r["train_rmsle"] = _float(text, r"Train RMSLE\s*[:\-]\s*([\d.]+)")
    r["test_rmsle"] = _float(text, r"Test\s+RMSLE\s*[:\-]\s*([\d.]+)")

    # Classification metrics
    r["train_acc"] = _float(text, r"Train Acc(?:uracy)?\s*[:\-]\s*([\d.]+)")
    r["test_acc"] = _float(text, r"Test\s+Acc(?:uracy)?\s*[:\-]\s*([\d.]+)")
    r["train_f1"] = _float(text, r"Train F1\s*[:\-]\s*([\d.]+)")
    r["test_f1"] = _float(text, r"Test\s+F1\s*[:\-]\s*([\d.]+)")
    r["train_auc"] = _float(text, r"Train AUC\s*[:\-]\s*([\d.]+)")
    r["test_auc"] = _float(text, r"Test\s+AUC\s*[:\-]\s*([\d.]+)")
    r["train_precision"] = _float(text, r"Train Precision\s*[:\-]\s*([\d.]+)")
    r["test_precision"] = _float(text, r"Test\s+Precision\s*[:\-]\s*([\d.]+)")
    r["train_recall"] = _float(text, r"Train Recall\s*[:\-]\s*([\d.]+)")
    r["test_recall"] = _float(text, r"Test\s+Recall\s*[:\-]\s*([\d.]+)")
    r["n_classes"] = _int(text, r"(\d+)\s+class(?:es)?", r"n.classes\s*[:\-]\s*(\d+)", r"num.classes\s*[:\-]\s*(\d+)")
    r["class_imbalance"] = _bool_present(text, r"imbalanc|class.weight|class.ratio|minority|majority", r"no.*imbalanc|balanced")
    r["imbalance_ratio"] = _float(text, r"imbalance.ratio\s*[:\-]\s*([\d.]+)", r"minority.*?([\d.]+)%", r"class.ratio.*?([\d.]+)")

    # CV
    r["cv_mean"] = _float(text, r"Mean Score\s*[:\-]\s*([\d.]+)", r"CV\s+Mean\s*[:\-]\s*([\d.]+)")
    r["cv_std"] = _float(text, r"Std\s+Score\s*[:\-]\s*([\d.]+)", r"CV\s+Std\s*[:\-]\s*([\d.]+)")
    r["cv_failed"] = bool(re.search(r"cv.*fail|fold.*fail|failed.*fold", text, re.I)
                          or (r["cv_mean"] is not None and r["cv_mean"] == 0.0
                              and r["cv_std"] is not None and r["cv_std"] == 0.0))

    # Target stats
    r["target_mean"] = _float(text, r"Target mean\s*[:\-]\s*([\d.]+)")
    r["target_std"] = _float(text, r"Target std\s*[:\-]\s*([\d.]+)")
    r["target_skewness"] = _float(text, r"Skewness\s*[:\-]\s*(-?[\d.]+)")

    # Data health
    r["n_duplicates"] = _int(text, r"(\d+)\s+duplicate rows")
    r["dup_pct"] = _float(text, r"duplicate rows.*?\((\d+\.?\d*)%", r"(\d+\.?\d*)\%.*?of data.*?duplicate")
    r["n_outliers"] = _int(text, r"Total outlier samples\s*[:\-]\s*(\d+)", r"(\d+)\s+outlier(?:\s+sample)?s")
    r["missing_vals"] = _bool_present(text, r"missing value|null value|NaN detected", r"No missing|0 missing")
    r["missing_pct"] = _float(text, r"missing.*?([\d.]+)%", r"null.*?([\d.]+)%")
    r["has_multicollin"] = _bool_present(text, r"highly correlated feature|multicollinear", r"No highly correlated")
    r["has_leakage"] = _bool_present(text, r"leakage|leak.*detected|suspiciously.*target", r"No.*leakage|no.*leak")
    r["has_low_variance"] = _bool_present(text, r"low.variance|constant feature|zero.variance", r"No.*low.variance|all.*reasonable")
    r["has_datetime"] = bool(re.search(r"date|datetime|timestamp|time.series|temporal", text, re.I))
    r["has_text_cols"] = bool(re.search(r"text.col|string.col|NLP|tfidf|tf.idf|embed", text, re.I))
    r["has_high_card"] = bool(re.search(r"high.cardinality|cardinality.*\d{3,}|unique.*\d{3,}", text, re.I))

    # Columns
    cols_m = re.search(r"COLUMNS\s*[:\-]\s*\[(.*?)\]", text, re.DOTALL)
    r["columns"] = re.findall(r"'(\w+)'", cols_m.group(1)) if cols_m else []

    r["target_col"] = _str(text, r"TARGET_COL\s*[:\-]\s*(\w+)", r"INFERRED_TARGET_COL\s*[:\-]\s*(\w+)", r"target column\s*[:\-]\s*'?(\w+)'?") or "target"
    r["n_cat_cols"] = _int(text, r"(\d+)\s+categorical col", r"Categorical cols\s*[:\-]\s*(\d+)") or 0

    # Feature importance (best-effort)
    feat_imp = {}
    perm = re.search(r"PERMUTATION IMPORTANCE.*?(?=STEP \d|$)", text, re.DOTALL | re.IGNORECASE)
    if perm:
        for m in re.finditer(r"(\w+)\s+([\d.]+)\s*\+?-?", perm.group()):
            feat_imp[m.group(1)] = float(m.group(2))
    if feat_imp and max(feat_imp.values()) > 1.5:
        total = sum(feat_imp.values())
        feat_imp = {k: v / total for k, v in feat_imp.items()}
    r["feature_importance"] = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

    # Model hyperparams
    r["depth"] = _int(text, r"depth\s*[:\-=]\s*(\d+)", r"max.depth\s*[:\-=]\s*(\d+)")
    r["lr"] = _float(text, r"learning.rate\s*[:\-=]\s*([\d.]+)")
    r["l2"] = _float(text, r"l2[_\s]leaf[_\s]reg\s*[:\-=]\s*([\d.]+)")
    r["iterations"] = _int(text, r"iterations\s*[:\-=]\s*(\d+)", r"n.estimators\s*[:\-=]\s*(\d+)", r"Trees built\s*[:\-]\s*(\d+)", r"n_trees\s*[:\-=]\s*(\d+)")
    r["best_iter"] = _int(text, r"Best iteration\s*[:\-]\s*(\d+)", r"best.round\s*[:\-]\s*(\d+)")

    if r["best_iter"] and r["iterations"]:
        ratio = r["best_iter"] / r["iterations"]
        r["early_stop_triggered"] = ratio < 0.92
        r["capacity_wasted"] = ratio < 0.6
        r["no_early_stop"] = ratio >= 0.99
    else:
        r["early_stop_triggered"] = False
        r["capacity_wasted"] = False
        r["no_early_stop"] = False

    KNOWN_BINARY_PATTERNS = re.compile(
        r"road|guest|basement|heating|air|cond|pref|area.*status|furnish|status|"
        r"gender|married|dependents|self.employ|education|loan|default|churn|fraud|"
        r"smoker|insured|incident|collision|bodily|authorities|auto|umbrella|capital",
        re.I,
    )
    r["numeric_features"] = [c for c in r["columns"] if c != r["target_col"] and not KNOWN_BINARY_PATTERNS.search(c)]
    return r

# =============================================================================
#  SECTION 2 - DIAGNOSIS ENGINE
# =============================================================================

def diagnose(r: dict) -> dict:
    issues = []
    warnings = []
    strengths = []
    summary_parts = []
    task = r.get("task", "regression")

    # Primary metric
    if task == "regression":
        train_score = r.get("train_r2")
        test_score = r.get("test_r2")
        metric_name = "R2"
        if test_score is not None and test_score < 0:
            issues.append(
                "CRITICAL: Test R2 is negative. Model performs worse than predicting the mean. "
                "Check leakage, contamination, or wrong target."
            )
    else:
        train_score = (r.get("train_f1") or r.get("train_auc") or r.get("train_acc"))
        test_score = (r.get("test_f1") or r.get("test_auc") or r.get("test_acc"))
        metric_name = "F1" if r.get("test_f1") else "AUC" if r.get("test_auc") else "Accuracy"

    # Gap
    reported_gap = r.get("r2_gap")
    if reported_gap is not None:
        gap = reported_gap
    elif train_score is not None and test_score is not None:
        gap = train_score - test_score
    else:
        gap = None

    # Overfitting
    overfit_level = "none"
    if gap is not None:
        if gap < -0.05:
            warnings.append(
                f"Test {metric_name} ({test_score:.3f}) is higher than train ({train_score:.3f}). "
                "Possible leakage or lucky split."
            )
        elif gap > 0.25:
            overfit_level = "severe"
            issues.append(
                f"Severe overfitting - {metric_name} gap {gap:.3f} (train={train_score:.3f}, test={test_score:.3f})."
            )
        elif gap > 0.12:
            overfit_level = "moderate"
            issues.append(f"Moderate overfitting - {metric_name} gap {gap:.3f}.")
        elif gap > 0.05:
            overfit_level = "mild"
            warnings.append(f"Mild overfitting - {metric_name} gap {gap:.3f}.")
        else:
            strengths.append(f"Good train/test alignment - {metric_name} gap {gap:.3f}.")
    else:
        warnings.append("Could not compute train/test gap - metrics not found.")

    # Underfitting
    underfit = False
    if test_score is not None and train_score is not None:
        if task == "regression" and train_score < 0.6 and test_score < 0.6:
            underfit = True
            issues.append(
                f"Underfitting - train R2={train_score:.3f}, test R2={test_score:.3f}. "
                "Increase capacity or add features."
            )
        elif task == "classification" and train_score < 0.70 and test_score < 0.70:
            underfit = True
            issues.append(
                f"Underfitting - train/test {metric_name} below 0.70. Add features or increase model capacity."
            )

    # Performance quality
    if test_score is not None and test_score >= 0 and not underfit:
        if task == "regression":
            if test_score >= 0.90:
                strengths.append(f"Excellent test R2 {test_score:.3f}.")
            elif test_score >= 0.80:
                strengths.append(f"Strong test R2 {test_score:.3f}.")
            elif test_score >= 0.70:
                warnings.append(f"Decent test R2 {test_score:.3f} - room to improve.")
            elif test_score >= 0.50:
                issues.append(f"Weak test R2 {test_score:.3f}.")
            else:
                issues.append(f"Very weak test R2 {test_score:.3f}. Investigate data quality.")
        else:
            if test_score >= 0.90:
                strengths.append(f"Excellent test {metric_name} {test_score:.3f}.")
            elif test_score >= 0.80:
                strengths.append(f"Strong test {metric_name} {test_score:.3f}.")
            elif 0.70 <= test_score < 0.80:
                warnings.append(f"Moderate test {metric_name} {test_score:.3f} - more work needed.")
            else:
                issues.append(f"Low test {metric_name} {test_score:.3f}.")

    # Data issues
    dupes = r.get("n_duplicates", 0) or 0
    dup_pct = r.get("dup_pct", 0) or 0
    if dupes > 0:
        (issues if dup_pct > 5 else warnings).append(
            f"{dupes} duplicate rows ({dup_pct:.1f}%). Deduplication required."
        )

    if r.get("missing_vals"):
        pct = r.get("missing_pct", "unknown")
        issues.append(f"Missing values detected ({pct}% missing). Imputation required.")

    if r.get("n_outliers", 0):
        warnings.append(f"{r['n_outliers']} outlier samples detected.")

    if r.get("has_multicollin"):
        warnings.append("High feature correlation detected. Consider dropping redundant features.")

    if r.get("has_leakage"):
        issues.append("Leakage warning detected in report. Remove suspect feature and retrain.")

    if r.get("has_low_variance"):
        warnings.append("Low-variance or constant features detected. Drop them.")

    if r.get("has_high_card"):
        warnings.append("High-cardinality categorical column detected. Use target or frequency encoding.")

    if r.get("has_datetime"):
        warnings.append("Datetime columns detected. Extract cyclical features and use time-aware CV.")

    if r.get("has_text_cols"):
        warnings.append("Text columns detected. Consider TF-IDF or embeddings.")

    if r.get("class_imbalance"):
        ratio = r.get("imbalance_ratio", "?")
        issues.append(
            f"Class imbalance detected (ratio ~{ratio}). Use class_weight='balanced' or resampling."
        )

    # Target distribution
    skew = r.get("target_skewness")
    if skew is not None:
        if skew > 0.75:
            issues.append(f"Right-skewed target (skewness={skew:.2f}). Apply log1p to target.")
        elif skew < -0.75:
            issues.append(
                f"Left-skewed target (skewness={skew:.2f}). Consider reflect-and-log transform."
            )
        else:
            strengths.append(f"Target skewness {skew:.2f} is mild.")

    # CV health
    if r.get("cv_failed"):
        issues.append("Cross-validation failed (all fold scores = 0). Fix CV setup first.")
    elif r.get("cv_std") is not None and r["cv_std"] > 0.05:
        warnings.append(f"High CV std {r['cv_std']:.3f}. Model unstable across folds.")

    # Iteration budget
    if r.get("no_early_stop") and r.get("iterations", 0) > 0:
        warnings.append(
            f"Model used all {r['iterations']} iterations. Configure early stopping."
        )
    if r.get("capacity_wasted"):
        best = r.get("best_iter", "?")
        total = r.get("iterations", "?")
        if isinstance(best, int) and isinstance(total, int) and total:
            pct = int(best / total * 100)
            warnings.append(
                f"Best iteration {best} was only {pct}% of max {total}. Reduce iterations."
            )

    # Model-specific notes
    model = (r.get("model") or "Unknown").lower()
    if "catboost" in model:
        if overfit_level in ("severe", "moderate"):
            issues.append("CatBoost depth and iterations appear over-parameterized.")
        strengths.append("CatBoost handles categoricals natively.")
    elif "lightgbm" in model or "lgbm" in model:
        if overfit_level in ("severe", "moderate"):
            issues.append("LightGBM overfitting. Increase min_child_samples and reduce num_leaves.")
    elif "xgboost" in model or "xgb" in model:
        if overfit_level in ("severe", "moderate"):
            issues.append("XGBoost overfitting. Increase min_child_weight and gamma.")
    elif "randomforest" in model:
        if overfit_level in ("severe", "moderate"):
            issues.append("RandomForest overfitting. Reduce depth and increase min_samples_leaf.")
    elif "linear" in model or "ridge" in model or "lasso" in model:
        if overfit_level in ("severe", "moderate"):
            issues.append("Linear model overfitting (rare). Increase regularization.")

    if r.get("gpu_hint"):
        strengths.append("GPU detected in report. Enable GPU training where supported.")

    # Feature importance summary
    fi = r.get("feature_importance", {})
    if fi:
        top = list(fi.keys())[:3]
        strengths.append(f"Top predictors: {', '.join(top)}.")
        dominated = [k for k, v in fi.items() if v < 0.005 and k != r.get("target_col")]
        if dominated:
            warnings.append(f"Near-zero importance: {', '.join(dominated[:5])}. Consider dropping.")
    else:
        warnings.append("No feature importance found in report.")

    # Summary
    n = r.get("n_samples", "?")
    summary_parts.append(f"Dataset: {n} samples, {r.get('n_features','?')} features, task={task}.")
    if underfit:
        summary_parts.append("PRIMARY PROBLEM: underfitting.")
    elif overfit_level in ("severe", "moderate"):
        gap_str = f"{gap:.2f}" if gap is not None else "?"
        summary_parts.append(f"PRIMARY PROBLEM: {overfit_level} overfitting (gap={gap_str}).")
    if test_score is not None and test_score >= 0:
        summary_parts.append(f"Test {metric_name}={test_score:.3f}.")
    if skew and abs(skew) > 0.75:
        summary_parts.append("Log-transform the target first.")

    return {
        "summary": " ".join(summary_parts),
        "issues": issues,
        "warnings": warnings,
        "strengths": strengths,
        "overfit_level": overfit_level,
        "underfit": underfit,
        "gap": gap,
        "test_score": test_score,
        "train_score": train_score,
        "metric_name": metric_name,
    }


# =============================================================================
#  SECTION 3 - FEATURE ENGINEERING ADVISOR
# =============================================================================

def recommend_features(r: dict) -> list[dict]:
    """
    Domain-agnostic feature engineering advisor.
    Works in two layers:
      A) Universal rules - fire based on data properties.
      B) Semantic rules  - detect column roles by name patterns.
    """
    cols = set(r.get("columns", []))
    target = r.get("target_col", "target")
    task = r.get("task", "regression")
    is_reg = task == "regression"
    is_clf = task == "classification"
    skew = r.get("target_skewness") or 0
    fi = r.get("feature_importance", {})
    low_imp = [k for k, v in fi.items() if v < 0.005 and k != target]
    underfit = r.get("underfit", False)

    feats = []

    def add(name, rationale, code, priority):
        feats.append({"name": name, "rationale": rationale, "code": code, "priority": priority})

    # ------------------------------
    # Layer A - Universal rules
    # ------------------------------

    # A1. Target transformation
    if is_reg and skew > 0.75:
        add(
            "log_transform_target",
            f"Right-skewed target (skewness={skew:.2f}). log1p reduces tail impact.",
            f"df['{target}'] = np.log1p(df['{target}'])  # inverse: np.expm1(predictions)",
            1,
        )
    elif is_reg and skew < -0.75:
        add(
            "reflect_log_target",
            f"Left-skewed target (skewness={skew:.2f}). Reflect then log-transform.",
            f"_max = df['{target}'].max() + 1\n"
            f"df['{target}'] = np.log1p(_max - df['{target}'])\n"
            f"# Inverse: predictions = _max - np.expm1(raw_pred)",
            1,
        )

    # A2. Missing values
    if r.get("missing_vals"):
        add(
            "impute_missing_values",
            "Missing values detected. Median for numeric; 'Unknown' for categoricals.",
            "from sklearn.impute import SimpleImputer\n"
            f"_num = df.select_dtypes(include='number').columns.drop(['{target}'], errors='ignore')\n"
            "_cat = df.select_dtypes(include=['object','category']).columns\n"
            "df[_num] = SimpleImputer(strategy='median').fit_transform(df[_num])\n"
            "df[_cat] = df[_cat].fillna('Unknown')",
            1,
        )

    # A3. Outlier winsorization
    if (r.get("n_outliers") or 0) > 0:
        add(
            "winsorize_outliers",
            f"{r['n_outliers']} outlier samples detected. Winsorize numeric features at 1st/99th percentile.",
            "from scipy.stats import mstats\n"
            f"_num = df.select_dtypes(include='number').columns.drop(['{target}'], errors='ignore')\n"
            "for _c in _num:\n"
            "    df[_c] = mstats.winsorize(df[_c], limits=[0.01, 0.01])",
            2,
        )

    # A4. Datetime cyclical encoding
    if r.get("has_datetime"):
        add(
            "datetime_cyclical_features",
            "Date/time columns detected. Extract cyclical month/day features.",
            "# Convert string dates first: df['col'] = pd.to_datetime(df['col'])\n"
            "_dcols = df.select_dtypes(include=['datetime64']).columns\n"
            "for _c in _dcols:\n"
            "    df[_c+'_year'] = df[_c].dt.year\n"
            "    df[_c+'_month_sin'] = np.sin(2*np.pi*df[_c].dt.month/12)\n"
            "    df[_c+'_month_cos'] = np.cos(2*np.pi*df[_c].dt.month/12)\n"
            "    df[_c+'_dow_sin'] = np.sin(2*np.pi*df[_c].dt.dayofweek/7)\n"
            "    df[_c+'_dow_cos'] = np.cos(2*np.pi*df[_c].dt.dayofweek/7)\n"
            "    df[_c+'_is_weekend'] = (df[_c].dt.dayofweek >= 5).astype(int)",
            2,
        )

    # A5. High-cardinality encoding
    if r.get("has_high_card"):
        add(
            "target_encode_high_cardinality",
            "High-cardinality categoricals detected. Use leave-one-out target encoding.",
            "# pip install category_encoders\n"
            "import category_encoders as ce\n"
            f"_hc = [c for c in df.select_dtypes('object').columns if df[c].nunique() > 20 and c != '{target}']\n"
            "if _hc:\n"
            "    _enc = ce.LeaveOneOutEncoder(cols=_hc, sigma=0.05)\n"
            f"    df[_hc] = _enc.fit_transform(df[_hc], df['{target}'])",
            3,
        )

    # A6. Aggregate binary flags
    non_target_cols = [c for c in cols if c != target]
    flag_cols = [
        c
        for c in non_target_cols
        if re.search(
            r"^(is|has|can|was|did|will|got|had)_?\w+|\w+_(flag|ind|yn|bin|bool|status|active|enabled|allowed)$",
            c,
            re.I,
        )
    ]
    if len(flag_cols) >= 3:
        add(
            "aggregate_flag_score",
            f"Found {len(flag_cols)} binary flag columns. Summing them can expose additive signals.",
            f"_flags = {flag_cols}\n"
            "for _c in _flags:\n"
            "    if df[_c].dtype == object:\n"
            "        df[_c+'_bin'] = df[_c].astype(str).str.lower().isin(['yes','true','1','y']).astype(int)\n"
            "_bin_cols = [c+'_bin' if df[c].dtype==object else c for c in _flags if c in df.columns]\n"
            "df['flag_score'] = df[[c for c in _bin_cols if c in df.columns]].sum(axis=1)",
            4,
        )

    # ------------------------------
    # Layer B - Semantic column roles
    # ------------------------------

    ROLE_PATTERNS = [
        (
            "id",
            r"^id$|_id$|^id_|customerid|userid|empid|rownum|"
            r"policy_number|account_number|order_id|transaction_id",
        ),
        (
            "flag",
            r"^(is|has|can|was|did|will|got|had)[_-]|"
            r"[_-](flag|ind|yn|bin|bool|active|enabled|allowed)$|"
            r"^(mainroad|basement|guestroom|hotwaterheating|airconditioning|"
            r"prefarea|default|churn|fraud|attrition|smoker|married|"
            r"graduate|self_employed|international|senior_citizen|"
            r"partner|dependents|paperless|overtime)$",
        ),
        (
            "size",
            r"\barea\b|sqft|sq_ft|\bsize\b|\bspace\b|\bfloor\b|footage|sqm|\bm2\b|"
            r"\bacreage\b|\bvolume\b|\bcapacity\b|dimension|length|width|height|depth|\bspan\b",
        ),
        (
            "count",
            r"\bnum_|\bn_|_count\b|\bqty\b|\bquantity\b|\btotal\b|\bnumber\b|no_of|"
            r"\brooms\b|\bbeds\b|\bbaths\b|\bfloors\b|\bseats\b|\bunits\b|\bmembers\b|"
            r"\bbedrooms?\b|\bbathrooms?\b|\bstories\b|\bdoors?\b|\bwindows?\b|\bemployees\b",
        ),
        (
            "rate",
            r"\brate\b|\bratio\b|\bpct\b|percent|\bscore\b|\bindex\b|\bgrade\b|"
            r"\brank\b|\brating\b|\bproportion\b|\bshare\b|\bfraction\b|\bgpa\b|\bbmi\b|\bdensity\b",
        ),
        (
            "duration",
            r"\bage\b|\btenure\b|\bduration\b|\bdays?\b|\bmonths?\b|\byears?\b|\bhours?\b|"
            r"\bperiod\b|\belapsed\b|\bsince\b|\bvintage\b|\bexperience\b|\bseniority\b|"
            r"contract_months?|loan_term",
        ),
        (
            "money",
            r"\bprice\b|\bcost\b|\bsalary\b|\bincome\b|\brevenue\b|\bamount\b|\bbalance\b|"
            r"\bpremium\b|\bfee\b|\bwage\b|\bpayment\b|\bspend\b|\bbudget\b|\bvalue\b|\bworth\b|"
            r"\bcredit\b|\bdebt\b|\bloan\b|\bcharge\b|\bbill\b|\btax\b|\bprofit\b|\bloss\b|\bmargin\b",
        ),
        (
            "distance",
            r"\bdist(ance)?\b|\bkm\b|\bmiles?\b|\blat\b|\blon\b|\bcoord\b|\bproximity\b|"
            r"\bradius\b|\bnearness\b|\blocation\b",
        ),
        (
            "category",
            r"\btype\b|\bcategory\b|\bclass\b|\bkind\b|\bbrand\b|\bproduct\b|\bsegment\b|"
            r"\btier\b|\bdepartment\b|\bteam\b|\bregion\b|\bcity\b|\bcountry\b|\bstate\b|\bzone\b|"
            r"\bgender\b|\bsex\b|\brace\b|\bethnicity\b|\beducation\b|\boccupation\b|"
            r"\bcontract\b|payment_method|internet_service|\bfurnishing\b",
        ),
    ]

    def get_role(col_name):
        for role, pat in ROLE_PATTERNS:
            if re.search(pat, col_name, re.I):
                return role
        return "unknown"

    role_map = {}
    by_role = {}
    for c in non_target_cols:
        role = get_role(c)
        role_map[c] = role
        by_role.setdefault(role, []).append(c)

    # B1. Ratio between size and count columns
    size_cols = by_role.get("size", [])
    count_cols = by_role.get("count", [])
    if size_cols and count_cols:
        s, c_ = size_cols[0], count_cols[0]
        add(
            f"{s}_per_{c_}",
            f"Ratio of size column '{s}' to count column '{c_}'.",
            f"df['{s}_per_{c_}'] = df['{s}'] / df['{c_}'].replace(0, 1)",
            3,
        )
        if len(size_cols) > 1:
            s2 = size_cols[1]
            add(
                f"{s2}_per_{c_}",
                f"Second size-per-count ratio: '{s2}' / '{c_}'.",
                f"df['{s2}_per_{c_}'] = df['{s2}'] / df['{c_}'].replace(0, 1)",
                3,
            )

    # B2. Product of size x count
    if size_cols and count_cols:
        s, c_ = size_cols[0], count_cols[0]
        add(
            f"{s}_x_{c_}",
            f"Product of '{s}' x '{c_}' to capture combined capacity.",
            f"df['{s}_x_{c_}'] = df['{s}'] * df['{c_}']",
            4,
        )

    # B3. Ratio between two count columns
    if len(count_cols) >= 2:
        a, b = count_cols[0], count_cols[1]
        add(
            f"{a}_to_{b}_ratio",
            f"Ratio between count columns '{a}' and '{b}'.",
            f"df['{a}_to_{b}_ratio'] = df['{a}'] / df['{b}'].replace(0, 1)",
            4,
        )

    # B4. Aggregate binary flag columns into a score
    flag_cols_sem = by_role.get("flag", [])
    if len(flag_cols_sem) >= 3 and len(flag_cols) < 3:
        add(
            "binary_flag_aggregate_score",
            f"Found {len(flag_cols_sem)} binary flag columns. Aggregate into a single score.",
            f"_fc = {flag_cols_sem}\n"
            "for _c in _fc:\n"
            "    if df[_c].dtype == object:\n"
            "        df[_c+'_bin'] = df[_c].astype(str).str.lower().isin(['yes','true','1','y']).astype(int)\n"
            "_bc = [c+'_bin' if df[c].dtype==object else c for c in _fc if c in df.columns]\n"
            "df['feature_score'] = df[[c for c in _bc if c in df.columns]].sum(axis=1)",
            3,
        )

    # B5. Duration / money interactions
    duration_cols = by_role.get("duration", [])
    money_cols = by_role.get("money", [])
    if duration_cols and money_cols:
        d, m = duration_cols[0], money_cols[0]
        add(
            f"{m}_per_{d}",
            f"Rate over time: '{m}' divided by '{d}'.",
            f"df['{m}_per_{d}'] = df['{m}'] / df['{d}'].replace(0, 1)",
            3,
        )

    # B6. Duration binning
    if duration_cols:
        d = duration_cols[0]
        add(
            f"{d}_bin",
            f"Quantile-bin '{d}' to capture non-linear effects.",
            f"df['{d}_bin'] = pd.qcut(df['{d}'], q=4, labels=['low','mid','high','very_high'], duplicates='drop').astype(str)",
            5,
        )

    # B7. Size/area binning
    if size_cols:
        s = size_cols[0]
        add(
            f"{s}_bin",
            f"Quantile-bin '{s}' to capture regime changes.",
            f"df['{s}_bin'] = pd.qcut(df['{s}'], q=5, labels=['xs','s','m','l','xl'], duplicates='drop').astype(str)",
            5,
        )

    # B8. Rate x money interaction
    rate_cols = by_role.get("rate", [])
    if rate_cols and money_cols:
        rt, mn = rate_cols[0], money_cols[0]
        add(
            f"{mn}_x_{rt}",
            f"Product of money column '{mn}' and rate/score column '{rt}'.",
            f"df['{mn}_x_{rt}'] = df['{mn}'] * df['{rt}']",
            4,
        )

    # B9. Distance interactions
    distance_cols = by_role.get("distance", [])
    if distance_cols and money_cols:
        dist, mn = distance_cols[0], money_cols[0]
        add(
            f"{mn}_per_{dist}",
            f"Value per distance: '{mn}' / '{dist}'.",
            f"df['{mn}_per_{dist}'] = df['{mn}'] / df['{dist}'].replace(0, 1)",
            4,
        )

    # B10. Drop ID columns
    id_cols = by_role.get("id", [])
    if id_cols:
        add(
            "drop_id_columns",
            f"ID columns detected: {id_cols}. Drop them to avoid leakage.",
            f"_id_cols = {id_cols}\n"
            "df.drop(columns=[c for c in _id_cols if c in df.columns], inplace=True, errors='ignore')",
            1,
        )

    # B11. Interaction between top-2 importance features
    fi_feats = [k for k in fi.keys() if k != target and k in cols]
    num_feats = r.get("numeric_features", [])
    top_num_fi = [f for f in fi_feats if f in num_feats][:4]
    if len(top_num_fi) >= 2:
        a, b = top_num_fi[0], top_num_fi[1]
        already_added = any(a in feat["name"] and b in feat["name"] for feat in feats)
        if not already_added:
            add(
                f"{a}_x_{b}_interaction",
                f"Interaction between top numeric features '{a}' and '{b}'.",
                f"df['{a}_x_{b}'] = df['{a}'] * df['{b}']",
                3,
            )
            add(
                f"{a}_div_{b}_ratio",
                f"Ratio of '{a}' to '{b}'.",
                f"df['{a}_div_{b}'] = df['{a}'] / df['{b}'].replace(0, df['{b}'].median())",
                4,
            )

    # B12. Classification - encode categoricals
    if is_clf:
        add(
            "encode_remaining_categoricals",
            "For classification, encode remaining categoricals before training.",
            "_obj = df.select_dtypes(include=['object','category']).columns.tolist()\n"
            f"_obj = [c for c in _obj if c != '{target}']\n"
            "if _obj:\n"
            "    from sklearn.preprocessing import OrdinalEncoder\n"
            "    _enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\n"
            "    df[_obj] = _enc.fit_transform(df[_obj])",
            6,
        )

    # ------------------------------
    # Layer C - Data-driven rules
    # ------------------------------

    if underfit or (r.get("n_cat_cols", 0) == 0 and len(num_feats) >= 2):
        add(
            "polynomial_features",
            "Add interaction-only polynomial features for underfitting or all-numeric data.",
            "from sklearn.preprocessing import PolynomialFeatures\n"
            f"_nc = df.select_dtypes(include='number').columns.drop(['{target}'], errors='ignore')\n"
            "_poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)\n"
            "_pa = _poly.fit_transform(df[_nc])\n"
            "_pn = _poly.get_feature_names_out(_nc)\n"
            "df = pd.concat([df, pd.DataFrame(_pa[:, len(_nc):], columns=_pn[len(_nc):], index=df.index)], axis=1)",
            6,
        )

    if low_imp:
        add(
            "drop_low_importance_features",
            f"Near-zero permutation importance features: {low_imp[:6]}.",
            f"_low = {low_imp}\n"
            "df.drop(columns=[c for c in _low if c in df.columns], inplace=True, errors='ignore')",
            7,
        )

    return sorted(feats, key=lambda x: x["priority"])


# =============================================================================
#  SECTION 4 - MODEL SELECTION ENGINE
# =============================================================================

def recommend_models(r: dict, diag: dict) -> list[dict]:
    task = r["task"]
    n = r.get("n_samples") or 0
    n_cat = r.get("n_cat_cols") or 0
    overfit_level = diag["overfit_level"]
    underfit = diag["underfit"]
    test_score = diag.get("test_score") or 0
    model_now = (r.get("model") or "Unknown").lower()
    is_reg = task == "regression"
    is_clf = task == "classification"
    imbalanced = r.get("class_imbalance", False)
    n_classes = r.get("n_classes") or 2
    has_datetime = r.get("has_datetime", False)
    gpu = r.get("gpu_hint", False)

    recs = []
    priority = [1]

    def next_p():
        p = priority[0]
        priority[0] += 1
        return p

    # LightGBM
    lgbm_hp = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31 if overfit_level in ("moderate", "severe") else 63,
        "min_child_samples": 30 if overfit_level in ("moderate", "severe") else 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "verbose": -1,
    }
    if is_clf and imbalanced:
        lgbm_hp["class_weight"] = "balanced"
        lgbm_hp["is_unbalance"] = True
    if is_clf and n_classes > 2:
        lgbm_hp["objective"] = "multiclass"
        lgbm_hp["num_class"] = n_classes
        lgbm_hp["metric"] = "multi_logloss"

    lgbm_rationale = "LightGBM provides strong regularization for tabular data."
    if "catboost" in model_now and overfit_level in ("moderate", "severe"):
        recs.append(
            {
                "model": "LGBM" + ("Regressor" if is_reg else "Classifier"),
                "priority": next_p(),
                "rationale": lgbm_rationale + " CatBoost is overfitting; try LGBM.",
                "hyperparameters": lgbm_hp,
            }
        )
    elif "lgbm" not in model_now and "lightgbm" not in model_now:
        recs.append(
            {
                "model": "LGBM" + ("Regressor" if is_reg else "Classifier"),
                "priority": next_p(),
                "rationale": lgbm_rationale + " Good default for tabular data.",
                "hyperparameters": lgbm_hp,
            }
        )

    # CatBoost
    cat_hp = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 4 if overfit_level in ("moderate", "severe") else 6,
        "l2_leaf_reg": 10 if overfit_level in ("moderate", "severe") else 3,
        "rsm": 0.8,
        "subsample": 0.8,
        "early_stopping_rounds": 50,
        "eval_metric": "RMSE" if is_reg else ("MultiClass" if n_classes > 2 else "AUC"),
        "allow_writing_files": False,
        "verbose": 100,
    }
    if n_cat > 0:
        cat_hp["# note"] = "Pass cat_features list to fit() - CatBoost handles them natively"
    if gpu:
        cat_hp["task_type"] = "GPU"
    recs.append(
        {
            "model": "CatBoost" + ("Regressor" if is_reg else "Classifier"),
            "priority": next_p(),
            "rationale": "CatBoost handles categoricals natively and is strong for tabular data.",
            "hyperparameters": cat_hp,
        }
    )

    # XGBoost
    xgb_hp = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 4 if overfit_level in ("moderate", "severe") else 6,
        "min_child_weight": 10 if overfit_level in ("moderate", "severe") else 3,
        "gamma": 0.2 if overfit_level in ("moderate", "severe") else 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "tree_method": "gpu_hist" if gpu else "hist",
        "early_stopping_rounds": 50,
    }
    if is_clf and imbalanced and n_classes == 2:
        xgb_hp["scale_pos_weight"] = r.get("imbalance_ratio", 1)
    if is_clf and n_classes > 2:
        xgb_hp["objective"] = "multi:softprob"
        xgb_hp["num_class"] = n_classes
    recs.append(
        {
            "model": "XGB" + ("Regressor" if is_reg else "Classifier"),
            "priority": next_p(),
            "rationale": "XGBoost with stronger regularization; good alternative to CatBoost.",
            "hyperparameters": xgb_hp,
        }
    )

    # RandomForest
    if overfit_level in ("moderate", "severe") or underfit:
        rf_hp = {
            "n_estimators": 500,
            "max_depth": 8 if not underfit else None,
            "min_samples_leaf": 5 if overfit_level in ("moderate", "severe") else 1,
            "max_features": 0.6,
            "n_jobs": -1,
            "random_state": 42,
        }
        if is_clf and imbalanced:
            rf_hp["class_weight"] = "balanced"
        recs.append(
            {
                "model": "RandomForest" + ("Regressor" if is_reg else "Classifier"),
                "priority": next_p(),
                "rationale": "Bagging ensemble to reduce variance.",
                "hyperparameters": rf_hp,
            }
        )

    # Linear baseline
    if n < 1000 or underfit:
        if is_reg:
            recs.append(
                {
                    "model": "Ridge + ElasticNet (Linear Baseline)",
                    "priority": next_p(),
                    "rationale": f"Only {n} samples; linear baselines can be strong and stable.",
                    "hyperparameters": {
                        "Ridge__alpha": "CV over [0.01,0.1,1,10,100]",
                        "ElasticNet__alpha": "0.01-10",
                        "ElasticNet__l1_ratio": "0.1-0.9",
                    },
                }
            )
        else:
            recs.append(
                {
                    "model": "LogisticRegression (L2 baseline)",
                    "priority": next_p(),
                    "rationale": "Interpretable baseline for classification.",
                    "hyperparameters": {
                        "C": "CV over [0.001,0.01,0.1,1,10]",
                        "class_weight": "balanced" if imbalanced else "None",
                        "solver": "lbfgs",
                        "max_iter": 1000,
                    },
                }
            )

    # Time-series aware
    if has_datetime:
        recs.append(
            {
                "model": "HistGradientBoostingRegressor / TimeSeriesSplit CV",
                "priority": next_p(),
                "rationale": "Date/time features detected. Use TimeSeriesSplit to avoid leakage.",
                "hyperparameters": {
                    "max_iter": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "l2_regularization": 1.0,
                    "early_stopping": True,
                },
            }
        )

    # Stacking (only if healthy)
    if test_score >= 0.65 and overfit_level in ("mild", "none") and not underfit:
        base = ["LGBMRegressor", "XGBRegressor"] if is_reg else ["LGBMClassifier", "XGBClassifier"]
        meta = "Ridge(alpha=1.0)" if is_reg else "LogisticRegression(C=1.0)"
        recs.append(
            {
                "model": "StackingEnsemble",
                "priority": next_p(),
                "rationale": "Stacking can capture residuals when base models are stable.",
                "hyperparameters": {"base_learners": base, "meta_learner": meta, "cv_folds": 5},
            }
        )

    # Large dataset alternative
    if n and n > 100_000:
        recs.append(
            {
                "model": "HistGradientBoostingRegressor (sklearn)",
                "priority": next_p(),
                "rationale": f"Large dataset ({n} rows). Histogram boosting is efficient.",
                "hyperparameters": {
                    "max_iter": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "l2_regularization": 1.0,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 20,
                },
            }
        )

    return sorted(recs, key=lambda x: x["priority"])


# =============================================================================
#  SECTION 5 - OVERFITTING / UNDERFITTING FIX ADVISOR
# =============================================================================

def recommend_fixes(r: dict, diag: dict) -> list[str]:
    level = diag["overfit_level"]
    underfit = diag["underfit"]
    task = r["task"]
    model = (r.get("model") or "Unknown").lower()
    n = r.get("n_samples") or 0
    skew = r.get("target_skewness") or 0
    fixes = []

    if underfit:
        fixes.append("UNDERFITTING FIXES:")
        fixes.append("- Add more features and interactions.")
        fixes.append("- Reduce regularization or increase model capacity.")
        fixes.append("- Switch to a more powerful model family.")
        fixes.append("- Confirm target definition and feature leakage.")
        return fixes

    if level == "none":
        return ["No significant overfitting. Keep monitoring."]

    if (r.get("n_duplicates") or 0) > 0:
        fixes.append("Step 0 - Deduplicate first: df.drop_duplicates().")

    if r.get("cv_failed"):
        fixes.append("Step 0 - Fix CV setup; current scores are invalid.")

    if task == "regression" and abs(skew) > 0.75:
        fixes.append("Step 1 - Log-transform the target (np.log1p).")

    fixes.append("Early stopping: add validation split and stop after no improvement.")

    if level in ("moderate", "severe"):
        fixes.append("Reduce tree depth (e.g., depth=4) and increase regularization.")
        fixes.append("Enable column subsampling (colsample_bytree=0.8).")
        fixes.append("Lower learning rate and increase iterations with early stopping.")

    if level == "severe":
        fixes.append("Row subsampling (subsample=0.7) and more regularization.")
        fixes.append("Run Optuna for 50-100 trials for better hyperparameters.")
        fixes.append("Drop least important features to reduce variance.")

    if n < 600:
        fixes.append(f"Small dataset ({n} samples): use 5-fold CV for stability.")

    if "randomforest" in model:
        fixes.append("RandomForest: set min_samples_leaf=5, max_features=0.6, max_depth=10.")
    if "linear" in model or "ridge" in model or "lasso" in model:
        fixes.append("Linear model overfitting: increase alpha or reduce feature space.")
    if r.get("class_imbalance"):
        fixes.append("Imbalanced classes: use class_weight='balanced' and evaluate F1/AUC.")
    if task == "classification" and (r.get("n_classes") or 2) > 2:
        fixes.append("Multiclass: use macro-averaged F1 as primary metric.")

    return fixes


# =============================================================================
#  SECTION 6 - TRAINING SCRIPT GENERATOR
# =============================================================================

def generate_script(r: dict, diag: dict, features: list, models: list, csv_path: str) -> str:
    task = r["task"]
    target = r.get("target_col", "target")
    is_reg = task == "regression"
    has_skew = abs(r.get("target_skewness") or 0) > 0.75 and is_reg
    has_dupes = (r.get("n_duplicates") or 0) > 0
    n_cat = r.get("n_cat_cols") or 0
    low_imp = [k for k, v in r.get("feature_importance", {}).items() if v < 0.005 and k != target]
    n = r.get("n_samples") or 0
    use_cv = n < 2000

    cat_cols_list = [
        c
        for c in r.get("columns", [])
        if re.search(
            r"mainroad|guestroom|basement|hotwaterheating|airconditioning|prefarea|furnishing|"
            r"road|heating|cond|status|gender|married|education|loan|churn|fraud|smoker|insured",
            c,
            re.I,
        )
        and c != target
    ]

    top = models[0] if models else {"model": "LGBMRegressor", "hyperparameters": {}}
    top_name = top["model"].lower()
    top_hp = top["hyperparameters"]

    if "lgbm" in top_name or "lightgbm" in top_name:
        lib = "lgbm"
    elif "catboost" in top_name:
        lib = "catboost"
    elif "xgb" in top_name or "xgboost" in top_name:
        lib = "xgb"
    elif "randomforest" in top_name:
        lib = "rf"
    elif "ridge" in top_name or "elastic" in top_name or "logistic" in top_name or "linear" in top_name:
        lib = "linear"
    else:
        lib = "lgbm"

    imports = [
        "import warnings; warnings.filterwarnings('ignore')",
        "import numpy as np",
        "import pandas as pd",
        "from pathlib import Path",
        "from sklearn.model_selection import train_test_split, KFold, StratifiedKFold",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
        if is_reg
        else "from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report",
        "import joblib",
    ]
    if lib == "lgbm":
        imports.append("import lightgbm as lgb")
    if lib == "catboost":
        imports.append("from catboost import CatBoostRegressor, CatBoostClassifier, Pool")
    if lib == "xgb":
        imports.append("import xgboost as xgb")
    if lib == "rf":
        imports.append("from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier")
    if lib == "linear" and is_reg:
        imports.append("from sklearn.linear_model import Ridge, ElasticNet")
    if lib == "linear" and not is_reg:
        imports.append("from sklearn.linear_model import LogisticRegression")

    # Feature engineering block
    fe_body = [
        "def engineer_features(df):",
        "    \"\"\"Apply all recommended feature engineering steps.\"\"\"",
        "    df = df.copy()",
    ]
    for feat in features:
        fe_body.append(f"\n    # {feat['name']}")
        fe_body.append(f"    # {feat['rationale'][:80]}")
        for line in feat["code"].split("\n"):
            fe_body.append(f"    {line}")
    fe_body.append("\n    return df")
    fe_block = "\n".join(fe_body)

    # Evaluation block
    if is_reg:
        eval_block = (
            "    mse = mean_squared_error(y_true, preds)\n"
            "    rmse = mse ** 0.5\n"
            "    mae = mean_absolute_error(y_true, preds)\n"
            "    r2 = r2_score(y_true, preds)\n"
            "    print(f'  RMSE : {rmse:>15,.2f}')\n"
            "    print(f'  MAE  : {mae:>15,.2f}')\n"
            "    print(f'  R2   : {r2:>15.4f}')\n"
            "    return {'rmse': rmse, 'mae': mae, 'r2': r2}"
        )
    else:
        eval_block = (
            "    acc = accuracy_score(y_true, preds.round())\n"
            "    f1 = f1_score(y_true, preds.round(), average='weighted')\n"
            "    print(f'  Accuracy : {acc:.4f}')\n"
            "    print(f'  F1       : {f1:.4f}')\n"
            "    print(classification_report(y_true, preds.round()))\n"
            "    return {'acc': acc, 'f1': f1}"
        )

    # Model fit block
    hp_kwargs = "\n".join(
        f"        {k}={repr(v)}," for k, v in top_hp.items() if not k.startswith("#") and k not in ("early_stopping_rounds",)
    )
    early_stop = top_hp.get("early_stopping_rounds", 50)

    if lib == "lgbm":
        model_cls = "lgb.LGBMRegressor" if is_reg else "lgb.LGBMClassifier"
        fit_code = f"""\
    model = {model_cls}(
{hp_kwargs}
        random_state=42, n_jobs=-1,
    )
    for col in cat_features:
        if col in X_tr.columns: X_tr[col] = X_tr[col].astype('category')
        if col in X_va.columns: X_va[col] = X_va[col].astype('category')
    model.fit(
        X_tr, y_tr_fit,
        eval_set=[(X_va, y_va_fit)],
        callbacks=[
            lgb.early_stopping({early_stop}, verbose=False),
            lgb.log_evaluation(200),
        ],
    )"""
    elif lib == "catboost":
        model_cls = "CatBoostRegressor" if is_reg else "CatBoostClassifier"
        fit_code = f"""\
    _cat = [c for c in cat_features if c in X_tr.columns]
    train_pool = Pool(X_tr, y_tr_fit, cat_features=_cat)
    val_pool = Pool(X_va, y_va_fit, cat_features=_cat)
    model = {model_cls}(
{hp_kwargs}
        random_seed=42,
    )
    model.fit(train_pool, eval_set=val_pool)"""
    elif lib == "xgb":
        model_cls = "xgb.XGBRegressor" if is_reg else "xgb.XGBClassifier"
        fit_code = f"""\
    from sklearn.preprocessing import OrdinalEncoder
    _cat = [c for c in cat_features if c in X_tr.columns]
    if _cat:
        _enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[_cat] = _enc.fit_transform(X_tr[_cat])
        X_va[_cat] = _enc.transform(X_va[_cat])
    model = {model_cls}(
{hp_kwargs}
        early_stopping_rounds={early_stop},
        random_state=42, n_jobs=-1, verbosity=1,
    )
    model.fit(X_tr, y_tr_fit, eval_set=[(X_va, y_va_fit)], verbose=200)"""
    elif lib == "rf":
        model_cls = "RandomForestRegressor" if is_reg else "RandomForestClassifier"
        fit_code = f"""\
    from sklearn.preprocessing import OrdinalEncoder
    _cat = [c for c in cat_features if c in X_tr.columns]
    if _cat:
        _enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[_cat] = _enc.fit_transform(X_tr[_cat])
        X_va[_cat] = _enc.transform(X_va[_cat])
    model = {model_cls}(
{hp_kwargs}
    )
    model.fit(X_tr, y_tr_fit)"""
    else:
        model_cls = "Ridge" if is_reg else "LogisticRegression"
        fit_code = f"""\
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    _cat = [c for c in cat_features if c in X_tr.columns]
    if _cat:
        _enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[_cat] = _enc.fit_transform(X_tr[_cat])
        X_va[_cat] = _enc.transform(X_va[_cat])
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_va = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)
    model = {model_cls}(alpha=1.0)
    model.fit(X_tr, y_tr_fit)"""

    pred_code = "np.expm1(model.predict(X_va))" if has_skew else "model.predict(X_va)"
    y_tr_fit = "np.log1p(y_tr)" if has_skew else "y_tr"
    y_va_fit = "np.log1p(y_va)" if has_skew else "y_va"

    fit_code_clean = textwrap.dedent(fit_code).strip("\n")

    metric_key = "r2" if is_reg else "f1"
    cv_class = "KFold" if is_reg else "StratifiedKFold"

    if use_cv:
        train_fn = f'''def run_training(X, y, cat_features):
    """5-fold cross-validation + final model on full train data."""
    kf = {cv_class}(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y if not {is_reg} else None), 1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        y_tr_fit = {y_tr_fit}
        y_va_fit = {y_va_fit}

        print(f"\\n-- Fold {{fold}}/5 --")
{chr(10).join("        "+l for l in fit_code_clean.split(chr(10)))}

        preds = {pred_code}
        metrics = evaluate(y_va, preds)
        fold_scores.append(metrics)

    scores = [s["{metric_key}"] for s in fold_scores]
    print(f"\\n{'='*60}")
    print(f"CV {metric_key.upper()}: {{np.mean(scores):.4f}} +/- {{np.std(scores):.4f}}")
    print(f"{'='*60}")

    return fold_scores'''
    else:
        train_fn = f'''def run_training(X, y, cat_features):
    """Single 80/20 train/test split."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    y_tr_fit = {y_tr_fit}
    y_va_fit = {y_va_fit}

    print("\\nTraining...")
{chr(10).join("    "+l for l in fit_code_clean.split(chr(10)))}

    preds = {pred_code}
    print("\\nTest Results:")
    metrics = evaluate(y_va, preds)

    joblib.dump(model, OUTPUT_DIR / "model.pkl")
    print(f"\\nModel saved -> {OUTPUT_DIR}/model.pkl")

    pd.DataFrame({{"actual": y_va.values, "predicted": preds}}).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False)
    print(f"Predictions -> {OUTPUT_DIR}/predictions.csv")

    return metrics'''

    if lib == "lgbm":
        optuna_objective = f'''\
    def objective(trial):
        params = {{
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbose": -1, "n_jobs": -1, "random_state": 42,
        }}
        model = lgb.{"LGBMRegressor" if is_reg else "LGBMClassifier"}(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_i, va_i in kf.split(X_opt):
            X_t, X_v = X_opt.iloc[tr_i], X_opt.iloc[va_i]
            y_t, y_v = y_opt.iloc[tr_i], y_opt.iloc[va_i]
            for col in cat_features_opt:
                if col in X_t.columns: X_t[col] = X_t[col].astype('category'); X_v[col] = X_v[col].astype('category')
            model.fit(X_t, {"np.log1p(y_t)" if has_skew else "y_t"},
                      eval_set=[(X_v, {"np.log1p(y_v)" if has_skew else "y_v"})],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            p = {"np.expm1(model.predict(X_v))" if has_skew else "model.predict(X_v)"}
            scores.append({"r2_score(y_v, p)" if is_reg else "f1_score(y_v, p.round(), average='weighted')"})
        return np.mean(scores)'''
    else:
        optuna_objective = '''\
    def objective(trial):
        import xgboost as xgb
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist", "n_jobs": -1, "random_state": 42,
        }
        from sklearn.model_selection import cross_val_score
        model = xgb.XGBRegressor(**params, verbosity=0)
        scores = cross_val_score(model, X_opt, y_opt, cv=3, scoring="r2")
        return scores.mean()'''

    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append('"""')
    lines.append("Auto-generated improved training script")
    lines.append("Generated by: ML Engineer Brain v2 (no API)")
    lines.append(f"Primary model: {top['model']}")
    lines.append(f"Task: {task}")
    lines.append(f"Features added: {len(features)}")
    lines.append('"""')
    lines.append("")
    lines.extend(imports)
    lines.append("")
    lines.append(f"CSV_PATH = r\"{csv_path}\"")
    lines.append(f"TARGET = \"{target}\"")
    lines.append("OUTPUT_DIR = Path(\"ml_brain_output\")")
    lines.append("OUTPUT_DIR.mkdir(exist_ok=True)")
    lines.append("RANDOM_STATE = 42")
    lines.append("")
    lines.append(f"CAT_FEATURES = {cat_cols_list}")
    lines.append("")
    lines.append("RUN_OPTUNA = False")
    lines.append("OPTUNA_TRIALS = 100")
    lines.append("")
    lines.append(fe_block)
    lines.append("")
    lines.append("")
    lines.append("def evaluate(y_true, preds):")
    lines.append(eval_block)
    lines.append("")
    lines.append("")
    lines.append("def load_data():")
    lines.append("    print(f\"Loading {CSV_PATH} ...\")")
    lines.append("    df = pd.read_csv(CSV_PATH)")
    lines.append("    print(f\"Raw shape: {df.shape}\")")
    lines.append("")
    if has_dupes:
        lines.append("    df.drop_duplicates(inplace=True)")
        lines.append("    print(f\"After dedup: {df.shape[0]} rows\")")
    else:
        lines.append("    # No duplicates reported")
    lines.append("")
    lines.append("    df = engineer_features(df)")
    lines.append("    print(f\"After feature engineering: {df.shape[1]} columns\")")
    lines.append("")
    lines.append("    cat_features = [c for c in df.select_dtypes(include=['object','category']).columns if c != TARGET]")
    lines.append("    for _c in cat_features:")
    lines.append("        df[_c] = df[_c].astype('category')")
    lines.append("")
    lines.append("    X = df.drop(columns=[TARGET])")
    lines.append("    y = df[TARGET]")
    lines.append("    return X, y, cat_features")
    lines.append("")
    lines.append("")
    lines.append(train_fn)
    lines.append("")
    lines.append("")
    lines.append("def run_optuna(X, y, cat_features):")
    lines.append("    try:")
    lines.append("        import optuna")
    lines.append("        optuna.logging.set_verbosity(optuna.logging.WARNING)")
    lines.append("    except Exception:")
    lines.append("        print(\"Install optuna: pip install optuna\")")
    lines.append("        return {}")
    lines.append("")
    lines.append("    X_opt = X")
    lines.append("    y_opt = y")
    lines.append("    cat_features_opt = cat_features")
    lines.append("")
    lines.append(optuna_objective)
    lines.append("")
    lines.append("    study = optuna.create_study(direction=\"maximize\")")
    lines.append("    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)")
    lines.append("    print(f\"\\nBest params: {study.best_params}\")")
    lines.append("    print(f\"Best score : {study.best_value:.4f}\")")
    lines.append("    return study.best_params")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    X, y, inferred_cat = load_data()")
    lines.append("    cat_features = [c for c in CAT_FEATURES if c in X.columns]")
    lines.append("    for _c in inferred_cat:")
    lines.append("        if _c not in cat_features:")
    lines.append("            cat_features.append(_c)")
    lines.append("")
    lines.append("    if RUN_OPTUNA:")
    lines.append("        print(f\"\\nRunning Optuna ({OPTUNA_TRIALS} trials)...\")")
    lines.append("        best = run_optuna(X, y, cat_features)")
    lines.append("        print(\"\\nRerun with these hyperparameters substituted above, then set RUN_OPTUNA=False\")")
    lines.append("    else:")
    lines.append("        run_training(X, y, cat_features)")

    script = "\n".join(lines)
    return script


# =============================================================================
#  SECTION 7 - PRETTY PRINTER
# =============================================================================

W = 80
BAR = "=" * W


def hdr(t):
    print(f"\n{BAR}\n  {t}\n{BAR}")


def print_diagnosis(diag):
    hdr("DIAGNOSIS")
    print(f"\n  {diag['summary']}\n")
    if diag["issues"]:
        print("  Issues:")
        for x in diag["issues"]:
            print("    - " + textwrap.fill(x, 74, subsequent_indent="      "))
    if diag["warnings"]:
        print("\n  Warnings:")
        for x in diag["warnings"]:
            print("    - " + textwrap.fill(x, 74, subsequent_indent="      "))
    if diag["strengths"]:
        print("\n  Strengths:")
        for x in diag["strengths"]:
            print(f"    - {x}")


def print_features(features):
    hdr("FEATURE ENGINEERING PLAN")
    for i, f in enumerate(features, 1):
        print(f"\n  [{i}] {f['name'].upper()}")
        print("      Why: " + textwrap.fill(f["rationale"], 72, subsequent_indent="           "))
        print("      Code:")
        for line in f["code"].split("\n"):
            print(f"          {line}")


def print_models(models):
    hdr("MODEL RECOMMENDATIONS")
    for m in models:
        print(f"\n  #{m['priority']}  {m['model']}")
        print("      Why: " + textwrap.fill(m["rationale"], 72, subsequent_indent="           "))
        if m["hyperparameters"]:
            print("      Hyperparameters:")
            for k, v in m["hyperparameters"].items():
                print(f"          {k}: {v}")


def print_fixes(fixes):
    hdr("FIXES (apply in order)")
    for i, f in enumerate(fixes, 1):
        print("  " + textwrap.fill(f"{i}. {f}", 76, subsequent_indent="     "))


# =============================================================================
#  SECTION 8 - FILE SAVER
# =============================================================================

def save_outputs(script, features, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "training_pipeline.py"
    train_path.write_text(script, encoding="utf-8")

    fe_lines = ["import pandas as pd\nimport numpy as np\n\n",
                "def engineer_features(df):\n    df = df.copy()\n"]
    for f in features:
        fe_lines.append(f"\n    # {f['name']}\n    # {f['rationale'][:80]}\n")
        for line in f["code"].split("\n"):
            fe_lines.append(f"    {line}\n")
    fe_lines.append("\n    return df\n")
    fe_path = out_dir / "feature_engineering.py"
    fe_path.write_text("".join(fe_lines), encoding="utf-8")

    return {"train_path": str(train_path), "fe_path": str(fe_path)}


# =============================================================================
#  SECTION 9 - PUBLIC API
# =============================================================================

def run_brain(report_text: str, csv_path: str, output_dir: Path | str) -> dict:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    report = report_text or ""
    r = parse_report(report)
    diag = diagnose(r)
    feats = recommend_features(r)
    models = recommend_models(r, diag)
    fixes = recommend_fixes(r, diag)
    script = generate_script(r, diag, feats, models, csv_path)
    paths = save_outputs(script, feats, output_dir)

    return {
        "diagnosis": diag,
        "features": feats,
        "models": models,
        "fixes": fixes,
        "paths": paths,
    }


# =============================================================================
#  SECTION 10 - CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="ML Engineer Brain v2 - hardcoded intelligence, no API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/ml_brain_v2.py --report output.txt --csv data.csv\n"
            "  python tools/ml_brain_v2.py --report output.txt --csv data.csv --apply\n"
            "  cat output.txt | python tools/ml_brain_v2.py --stdin --csv data.csv"
        ),
    )
    p.add_argument("--report", help="Path to pipeline output report (.txt)")
    p.add_argument("--stdin", action="store_true", help="Read report from stdin")
    p.add_argument("--csv", default="data.csv", help="Path to the dataset CSV")
    p.add_argument("--output-dir", default="ml_brain_output")
    p.add_argument("--apply", action="store_true", help="Run the generated script immediately")
    args = p.parse_args()

    if args.stdin:
        report = sys.stdin.read()
        print("Read report from stdin")
    elif args.report:
        rp = Path(args.report)
        if not rp.exists():
            print(f"Not found: {rp}")
            sys.exit(1)
        report = rp.read_text(encoding="utf-8", errors="replace")
        print(f"Loaded: {rp} ({len(report):,} chars)")
    else:
        p.print_help()
        sys.exit(1)

    t0 = time.time()
    print("Parsing...")
    r = parse_report(report)
    print("Diagnosing...")
    diag = diagnose(r)
    print("Planning features...")
    feats = recommend_features(r)
    print("Selecting models...")
    models = recommend_models(r, diag)
    print("Computing fixes...")
    fixes = recommend_fixes(r, diag)
    print("Generating script...")
    script = generate_script(r, diag, feats, models, args.csv)
    print(f"Done in {time.time() - t0:.2f}s")

    print_diagnosis(diag)
    print_features(feats)
    print_models(models)
    print_fixes(fixes)

    hdr("GENERATED FILES")
    out_dir = Path(args.output_dir)
    save_outputs(script, feats, out_dir)

    if args.apply:
        import subprocess
        sp = out_dir / "training_pipeline.py"
        print(f"Running {sp} ...")
        proc = subprocess.run([sys.executable, str(sp)])
        if proc.returncode != 0:
            print(f"Exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
