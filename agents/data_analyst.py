"""
data_analyst.py — Data Analyst Agent

Responsibilities:
  - Twice-daily pipeline health checks (9AM + 9PM UTC)
  - Write timestamped rows to monitoring Excel file (openpyxl)
  - Monthly report compilation
  - Translate raw metrics into business-readable summaries
  - Flag anomalies to ML Engineer when detected
  - Track all incidents in the Excel incident log

Phase 1: All logic mocked.
Phase 2: Wire up openpyxl + APScheduler + real metrics from MLflow.
"""
import asyncio
from datetime import datetime
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a Data Analyst agent on an autonomous AI engineering platform.

Your responsibilities:
1. Run twice-daily health checks (9AM and 9PM UTC):
   - Pull model accuracy, drift score, and latency from MLflow
   - Check data pipeline status (last run timestamp, row count, schema changes)
   - Write a timestamped summary row to monitoring.xlsx
   - If any metric is outside threshold, flag immediately

2. Thresholds to monitor:
   - Model accuracy: alert if drops >2% from baseline
   - Drift score: alert if >0.10 (healthy range: 0.00–0.09)
   - Inference latency p95: alert if >200ms
   - Data freshness: alert if last ingest >4h ago
   - Null rates: alert if any feature exceeds 15%

3. Monthly report (1st of each month):
   - Model performance trend (30-day rolling accuracy)
   - Incident log summary (count, avg resolution time)
   - Security finding summary (from SAST monthly report)
   - Infrastructure health summary
   - Delivered as Excel attachment in chat

4. Translate findings into plain English for non-technical stakeholders.
   Never use raw numbers without context.

Tools available (Phase 2+):
  - openpyxl_tool: read/write Excel files
  - mlflow_tool: query experiment and model metrics
  - sql_tool: query monitoring database
  - plotly_tool: generate charts for reports

Communication:
  - Flag drift/accuracy issues to ML Engineer: message("ml_engineer", ...)
  - Send data quality incidents to Data Scientist: message("data_scientist", ...)
  - Report findings to orchestrator via report()
"""


# Simulated current metrics (Phase 1 mock)
MOCK_METRICS = {
    "model_accuracy": 94.1,
    "drift_score": 0.03,
    "latency_p50_ms": 42,
    "latency_p95_ms": 118,
    "inference_volume_today": 8420,
    "data_freshness_hours": 1.2,
    "last_check": "09:00 UTC",
}

THRESHOLDS = {
    "model_accuracy_drop": 2.0,   # percentage points
    "drift_score_max": 0.10,
    "latency_p95_max_ms": 200,
    "data_freshness_max_hours": 4,
}


class DataAnalystAgent(BaseAgent):
    AGENT_ID = "data_analyst"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        c = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        # Data Scientist asked about historical null rates
        if from_agent == "data_scientist" and any(kw in c for kw in ["null", "history", "log", "trend"]):
            await self.report(
                "Checked historical monitoring logs:\n"
                "  promo_clicks null rate history:\n"
                "    Feb 01: 2.1% (normal)\n"
                "    Feb 05: 2.3% (normal)\n"
                "    Feb 08: 38.4% ⚠ (spike begins)\n"
                "    Feb 09: 41.2% ⚠ (current)\n"
                "  Root cause: tracking pixel outage starting Feb 8th.\n"
                "  Logging as data quality incident DQI-2024-012.",
                task_id
            )
            return None

        # Scheduled health check (triggered by APScheduler in Phase 5)
        if any(kw in c for kw in ["health", "check", "scheduled", "9am", "9pm", "monitor"]):
            await self._run_health_check(task_id)
            return None

        # ML Engineer / someone asking for current metrics
        if any(kw in c for kw in ["metrics", "report", "status", "kpi", "dashboard", "performance"]):
            await self.report(
                f"Current pipeline metrics ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}):\n"
                f"  Model accuracy: {MOCK_METRICS['model_accuracy']}% (baseline: 92.0%)\n"
                f"  Drift score:    {MOCK_METRICS['drift_score']} (threshold: {THRESHOLDS['drift_score_max']})\n"
                f"  Latency p50:    {MOCK_METRICS['latency_p50_ms']}ms\n"
                f"  Latency p95:    {MOCK_METRICS['latency_p95_ms']}ms "
                f"(threshold: {THRESHOLDS['latency_p95_max_ms']}ms)\n"
                f"  Volume today:   {MOCK_METRICS['inference_volume_today']:,} predictions\n"
                f"  Data freshness: {MOCK_METRICS['data_freshness_hours']}h ago\n"
                f"  Last check:     {MOCK_METRICS['last_check']}",
                task_id
            )
            return None

        # Drift / incident flagged externally
        if any(kw in c for kw in ["drift", "incident", "drop", "degraded", "issue", "alert"]):
            await self.report(
                "Investigating anomaly in monitoring logs...",
                task_id
            )
            await asyncio.sleep(0.3)
            await self.report(
                "Incident INC-2024-007 opened:\n"
                "  Type: Model accuracy degradation\n"
                "  Detected: 2024-02-10 14:32 UTC\n"
                "  Severity: HIGH\n"
                "  Accuracy drop: 94.1% → 90.9% (-3.2%)\n"
                "  Correlated feature: age distribution shift\n"
                "  Logging to monitoring.xlsx and alerting ML Engineer.",
                task_id
            )
            await self.message(
                "ml_engineer",
                "Incident INC-2024-007: model accuracy dropped 3.2% over 48h. "
                "Correlates with `age` feature distribution shift. "
                "Please investigate and retrain.",
                task_id
            )
            return None

        # ML Engineer deployed — update logs
        if from_agent == "ml_engineer" and any(kw in c for kw in ["deployed", "live", "complete", "ci passed"]):
            await self.report(
                "Monitoring report updated:\n"
                "  Event: Model deployment\n"
                "  Timestamp: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "\n"
                "  Model accuracy logged: 94.1%\n"
                "  Row added to monitoring.xlsx (sheet: Pipeline Events)\n"
                "  Next scheduled health check: 9PM UTC",
                task_id
            )
            return None

        # Default
        await self.report(
            "Data Analyst ready.\n"
            "  Last health check: 9AM UTC — all metrics within thresholds.\n"
            "  No active incidents.\n"
            "  Next check: 9PM UTC.\n"
            "  Monthly report: scheduled for the 1st.",
            task_id
        )
        return None

    async def _run_health_check(self, task_id: str) -> None:
        """
        Full health check routine — runs at 9AM and 9PM UTC.
        Phase 5: triggered by APScheduler job.
        Phase 2: pulls real metrics from MLflow + monitoring DB.
        """
        await self.report("Running scheduled health check...", task_id)
        await asyncio.sleep(0.3)

        # Check each metric against thresholds
        issues = []

        if MOCK_METRICS["drift_score"] > THRESHOLDS["drift_score_max"]:
            issues.append(f"DRIFT ⚠ score {MOCK_METRICS['drift_score']} > threshold {THRESHOLDS['drift_score_max']}")

        if MOCK_METRICS["latency_p95_ms"] > THRESHOLDS["latency_p95_max_ms"]:
            issues.append(f"LATENCY ⚠ p95={MOCK_METRICS['latency_p95_ms']}ms > threshold {THRESHOLDS['latency_p95_max_ms']}ms")

        if MOCK_METRICS["data_freshness_hours"] > THRESHOLDS["data_freshness_max_hours"]:
            issues.append(f"DATA FRESHNESS ⚠ last ingest {MOCK_METRICS['data_freshness_hours']}h ago")

        if issues:
            await self.report(
                "Health check — ISSUES FOUND:\n" +
                "\n".join(f"  - {i}" for i in issues),
                task_id
            )
            await self.message("ml_engineer", f"Health check found issues: {'; '.join(issues)}", task_id)
        else:
            await self.report(
                f"Health check — ALL CLEAR ✅\n"
                f"  Accuracy: {MOCK_METRICS['model_accuracy']}% | "
                f"Drift: {MOCK_METRICS['drift_score']} | "
                f"Latency p95: {MOCK_METRICS['latency_p95_ms']}ms | "
                f"Volume: {MOCK_METRICS['inference_volume_today']:,}\n"
                f"  Row written to monitoring.xlsx.",
                task_id
            )

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.excel_tools import openpyxl_read_tool, openpyxl_write_tool
        # from tools.ml_tools import mlflow_metrics_tool
        # return [openpyxl_read_tool, openpyxl_write_tool, mlflow_metrics_tool]