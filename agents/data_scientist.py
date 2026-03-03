"""
data_scientist.py — Data Scientist Agent

Responsibilities:
  - Exploratory data analysis (EDA): nulls, distributions, correlations, outliers
  - Hypothesis testing and statistical significance
  - Feature engineering recommendations
  - Experiment design (A/B tests, holdout splits, significance thresholds)
  - Validates retrained models before they go to SAST/deploy

Phase 1: All logic mocked.
Phase 2: Wire up Pandas + SciPy + Evidently + Matplotlib.
"""
import asyncio
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a senior Data Scientist agent on an autonomous AI engineering platform.

Your responsibilities:
1. EDA checklist for every dataset:
   - Shape and dtypes
   - Null/missing rates per column (flag anything >10%)
   - Distribution summary: mean, std, skew, kurtosis
   - Correlation matrix: flag multicollinearity >0.85
   - Outlier detection (IQR method, flag >3 IQR)
   - Target variable distribution and class balance check

2. Statistical experiments:
   - Design A/B tests with proper control/variant splits (stratified by target)
   - Calculate required sample sizes for statistical power (default: 80% power, α=0.05)
   - Run t-tests, chi-square, Mann-Whitney as appropriate
   - Report: effect size, p-value, confidence intervals

3. Feature engineering:
   - Log/quantile transforms for skewed features
   - One-hot / ordinal encoding for categoricals
   - Interaction terms if domain knowledge warrants
   - Feature importance analysis post-training

4. Model validation:
   - Check feature importance stability across runs
   - Holdout set performance (no data leakage check)
   - Cross-validation results
   - Flag if any metric degrades vs. baseline

Tools available (Phase 2+):
  - pandas_tool: load, inspect, transform dataframes
  - scipy_tool: statistical tests, distributions
  - evidently_tool: drift detection, data quality reports
  - matplotlib_tool: EDA plots as base64 PNGs

Communication:
  - Send feature recommendations to ML Engineer after EDA
  - Flag data quality incidents to Data Analyst for logging
  - Report findings to orchestrator on task completion
"""


class DataScientistAgent(BaseAgent):
    AGENT_ID = "data_scientist"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        # Data Analyst confirmed data quality issue
        if from_agent == "data_analyst" and "null" in c:
            await self.report(
                "Data Analyst confirmed: promo_clicks null issue started Feb 8th. "
                "Safe to drop permanently from feature set.",
                task_id
            )
            await self.message(
                "ml_engineer",
                "Confirmed: drop promo_clicks from pipeline. "
                "Data Analyst confirmed the source is permanently broken.",
                task_id
            )
            return None

        # ML Engineer asked for validation of retrained model
        if from_agent == "ml_engineer" and any(kw in c for kw in ["validate", "check", "verify", "retrained"]):
            await self.report(
                "Model validation complete:\n"
                "  Feature importances: consistent with baseline (no unexpected shifts)\n"
                "  Top-3 unchanged: session_duration, page_depth, account_age\n"
                "  Accuracy gain source: quantile transform on account_age (+1.7%)\n"
                "  Holdout set: 95.1% — no overfitting detected\n"
                "  Cross-val (5-fold): 94.8% ± 0.4%\n"
                "Recommend proceeding to SAST and deployment.",
                task_id
            )
            return None

        # EDA request
        if any(kw in c for kw in ["eda", "explore", "analysis", "feature", "dataset", "data"]):
            await self.report("EDA started. Loading dataset and profiling...", task_id)
            await asyncio.sleep(0.3)
            await self.report(
                "EDA complete:\n"
                "  Shape: 45,231 rows × 24 columns\n"
                "  Nulls: promo_clicks (41%) ⚠, last_login (12%), referral_code (8%)\n"
                "  Skew: account_age (2.3) → log-transform recommended\n"
                "       session_duration (1.8) → log-transform recommended\n"
                "  Multicollinearity: age ↔ tenure (r=0.91) → drop tenure\n"
                "  Class imbalance: 78% / 22% → consider SMOTE or class_weight\n"
                "  Outliers: 0.3% extreme values in account_balance (IQR method)",
                task_id
            )

            # Flag quality issue to Data Analyst
            await self.message(
                "data_analyst",
                "Data quality incident: promo_clicks null rate at 41%. "
                "Please check historical monitoring logs for when this started.",
                task_id
            )

            # Send recommendations to ML Engineer
            await self.message(
                "ml_engineer",
                "EDA complete. Feature recommendations:\n"
                "  Drop: promo_clicks (broken), referral_code (>40% null), tenure (collinear)\n"
                "  Log-transform: account_age, session_duration\n"
                "  One-hot encode: region (7 categories), device_type (4 categories)\n"
                "  Add class_weight='balanced' to RandomForest",
                task_id
            )
            return None

        # Experiment design
        if any(kw in c for kw in ["experiment", "ab", "a/b", "hypothesis", "test"]):
            await self.report(
                "Experiment EXP-2024-031 designed:\n"
                "  Hypothesis: quantile-transform on account_age outperforms log-transform\n"
                "  Control: log-transform (current baseline)\n"
                "  Variant: quantile-transform (100 quantiles)\n"
                "  Split: 70/30 stratified by target label\n"
                "  Success metric: validation accuracy (min detectable effect: 0.5%)\n"
                "  Required sample: ~8,000 per arm (power=0.80, α=0.05)\n"
                "  Duration: single batch test (no time component)",
                task_id
            )
            await self.message(
                "ml_engineer",
                "Experiment EXP-2024-031 designed. "
                "Please implement both variants and run them in the E2B sandbox.",
                task_id
            )
            return None

        # Default status
        await self.report(
            "Data Scientist ready.\n"
            "  Last experiment: EXP-2024-031 (quantile vs log-transform) — "
            "Variant B won by +2.1%, promoted to main pipeline.\n"
            "  Dataset freshness: last updated 2h ago.\n"
            "  No active quality incidents.",
            task_id
        )
        return None

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.data_tools import pandas_tool, scipy_tool, evidently_tool
        # return [pandas_tool, scipy_tool, evidently_tool]