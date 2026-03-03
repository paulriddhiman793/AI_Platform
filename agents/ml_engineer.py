"""
ml_engineer.py — ML Engineer Agent

Responsibilities:
  - Write complete ML pipelines: data loading → preprocessing → training → evaluation → deploy
  - Self-healing debug loop: write → run in E2B → error? → fix (max 5x) → deploy
  - Request SAST scan before every GitHub push (hard rule)
  - Track experiments in MLflow
  - Set up APScheduler twice-daily monitoring jobs
  - First responder when Data Analyst flags model drift

Phase 1: All logic mocked with scripted responses.
Phase 2: Wire up E2B sandbox + MLflow + PyGitHub + Claude API.
"""
import asyncio
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a senior ML Engineer agent on an autonomous AI engineering platform.

Your responsibilities:
1. Write complete ML pipelines from scratch (sklearn, PyTorch, or TensorFlow depending on task).
2. Always use the self-healing loop: write code → run in E2B sandbox → if error, fix and retry (max 5x).
3. Before committing ANY code to GitHub, send it to SAST for review and wait for approval.
4. Log all experiments to MLflow: parameters, metrics, model artifacts.
5. Set up twice-daily monitoring cron jobs (9AM + 9PM UTC) using APScheduler.
6. When Data Analyst reports drift or accuracy drop, investigate and fix immediately.
7. Commit only to feature branches. Never push directly to main.

Tools available (Phase 2+):
  - code_execution: run Python in E2B cloud sandbox
  - github_tools: create branch, commit, push, open PR, monitor CI run
  - mlflow_tools: log experiment, register model, compare runs
  - sklearn_tools: fit, predict, evaluate (accuracy, F1, AUC, confusion matrix)

Self-healing loop (ALWAYS follow this):
  for attempt in range(1, 6):
      result = execute_in_sandbox(code)
      if result.success:
          break
      code = llm_fix(code, result.error)
  if not result.success:
      escalate_to_user(code, error, attempts_made=5)

SAST rule: Never skip the SAST scan. If SAST finds issues, fix them before deploying.
"""


class MLEngineerAgent(BaseAgent):
    AGENT_ID = "ml_engineer"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        from_agent = payload.get("from", "orchestrator")
        content_lower = content.lower()

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────

        # SAST gave approval — proceed to deploy
        if from_agent == "sast" and any(kw in content_lower for kw in ["approved", "clean", "pass"]):
            await self.report("SAST approval received. Committing to GitHub...", task_id)
            await asyncio.sleep(0.5)
            await self.report("Committed to feature/model-pipeline. CI run started. Monitoring...", task_id)
            await asyncio.sleep(0.8)
            await self.report(
                "CI passed ✅. Model deployed to production. "
                "MLflow entry logged. Accuracy: 94.1%.",
                task_id
            )
            return None

        # SAST found issues — apply fix and resubmit
        if from_agent == "sast" and any(kw in content_lower for kw in ["issue", "found", "fix", "high", "critical"]):
            await self.report(
                "SAST findings received. Applying fixes now...",
                task_id
            )
            await asyncio.sleep(0.5)
            await self.report(
                "Fixes applied:\n"
                "  - Replaced hardcoded DB string with os.getenv('DB_CONN_STR')\n"
                "  - Added path sanitization with os.path.abspath()\n"
                "Re-submitting to SAST for re-scan.",
                task_id
            )
            await self.message("sast", "Fixes applied. Please re-scan before I push.", task_id)
            return None

        # Data Scientist sent EDA recommendations
        if from_agent == "data_scientist" and any(kw in content_lower for kw in ["eda", "feature", "recommend"]):
            await self.report(
                "EDA recommendations received from Data Scientist. "
                "Updating pipeline:\n"
                "  - Dropping: promo_clicks, referral_code\n"
                "  - Log-transforming: account_age, session_duration\n"
                "  - One-hot encoding: region, device_type\n"
                "  - Dropping: tenure (collinear with age)",
                task_id
            )
            return None

        # Data Analyst flagged drift
        if from_agent == "data_analyst" and any(kw in content_lower for kw in ["drift", "drop", "degraded", "accuracy"]):
            await self.report(
                "Drift alert received. Pulling last 48h data slice for analysis...",
                task_id
            )
            await asyncio.sleep(0.5)
            await self.report(
                "Root cause identified: distribution shift in `age` feature — "
                "new 18-22 cohort not in training data. "
                "Retraining with updated distribution now.",
                task_id
            )
            await asyncio.sleep(0.8)
            await self.report(
                "Retraining complete. Accuracy restored to 94.8% "
                "(slight improvement from better cohort coverage). "
                "Sending to SAST before deploy.",
                task_id
            )
            await self.message(
                "sast",
                "Drift fix ready. Retrained model deploy script — "
                "please scan before I push.",
                task_id
            )
            return None

        # Default: build a new ML pipeline
        if any(kw in content_lower for kw in ["model", "train", "pipeline", "build", "churn", "predict"]):
            await self.report(
                "Starting pipeline build.\n"
                "  Dataset: loading from /data/...\n"
                "  Shape: 12,450 rows × 18 features\n"
                "  Baseline model: RandomForestClassifier",
                task_id
            )
            await asyncio.sleep(0.4)

            # Ask Data Scientist for EDA before finalizing
            await self.message(
                "data_scientist",
                "I'm building the ML pipeline. Can you run EDA and flag anything "
                "before I finalize feature selection?",
                task_id
            )

            await asyncio.sleep(0.3)
            await self.report(
                "Pipeline skeleton written. Running in E2B sandbox...",
                task_id
            )

            # Run self-healing loop (mocked — always succeeds)
            result = await self._run_with_healing(
                code="# RandomForest training pipeline\n# (mock code)",
                task_id=task_id
            )

            if result["success"]:
                await self.report(
                    f"Sandbox run succeeded on attempt {result['attempts']}. "
                    "Training complete. Accuracy: 94.1% | F1: 0.89 | AUC: 0.96.\n"
                    "Requesting SAST scan before deploy.",
                    task_id
                )
                await self.message(
                    "sast",
                    "ML pipeline script ready for deployment. "
                    "Please run security scan before I push to GitHub.",
                    task_id
                )
            else:
                await self.report(
                    "Could not auto-fix after 5 attempts. "
                    "Escalating to user with full error context.",
                    task_id
                )
            return None

        # Generic status check
        await self.report(
            "ML Engineer ready.\n"
            "  Last run: 9AM health check — all clear\n"
            "  Active model: churn_v3 (accuracy: 94.1%)\n"
            "  Drift score: 0.03 (healthy, threshold 0.10)\n"
            "  Next scheduled check: 9PM UTC",
            task_id
        )
        return None

    def get_tools(self):
        return []
        # Phase 2+:
        # from tools.code_execution import e2b_run_tool
        # from tools.github_tools import github_commit_tool, github_pr_tool
        # from tools.ml_tools import mlflow_log_tool, mlflow_register_tool
        # return [e2b_run_tool, github_commit_tool, github_pr_tool, mlflow_log_tool]