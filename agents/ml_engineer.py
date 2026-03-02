"""
ml_engineer.py — ML Engineer Agent

Responsibilities:
  - Build, train, evaluate, and deploy ML pipelines
  - Self-healing debug loop (write → run → error? → fix → retry × 5)
  - Coordinate with SAST before pushing code to GitHub
  - Set up twice-daily monitoring cron jobs
  - First responder when Data Analyst flags model drift

Phase 1: All LLM calls and tool calls are MOCKED.
Phase 2: Wire up real Claude API + E2B sandbox + MLflow + GitHub tools.
"""
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a senior ML Engineer agent on an autonomous AI engineering platform.

Your responsibilities:
1. Write complete, runnable ML pipelines (data loading → preprocessing → training → evaluation → deployment).
2. Use the self-healing debug loop: write code → execute in E2B sandbox → if error, diagnose and fix → retry (max 5x).
3. Before pushing ANY code to GitHub, you MUST request a SAST scan and wait for approval.
4. Track all experiments in MLflow. Commit trained models to the model registry.
5. Set up APScheduler cron jobs for twice-daily monitoring (9AM + 9PM UTC).
6. When the Data Analyst flags drift or performance degradation, you are the first responder.

Tools available (Phase 2+):
  - code_execution: runs Python code in E2B cloud sandbox
  - github_tools: commit, push, PR creation, CI monitoring
  - mlflow_tools: experiment logging, model registration
  - evidently_tools: drift detection and data quality checks

Communication:
  - Send code to SAST for review before deploying: message("sast", "Please scan: <code>")
  - Request EDA from Data Scientist: message("data_scientist", "Please run EDA on: <dataset>")
  - Report completion to orchestrator via report()

Self-healing loop (max retries = 5):
  for attempt in range(5):
      result = execute_in_sandbox(code)
      if result.success: break
      code = fix_code(code, result.error)
  if not result.success:
      escalate_to_user(code, result.error, attempts)
"""


class MLEngineerAgent(BaseAgent):
    AGENT_ID = "ml_engineer"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    async def handle_task(self, payload: dict) -> str:
        content = payload.get("content", "")
        task_id = payload.get("task_id")

        # ── PHASE 1 MOCK RESPONSES ──────────────────────────────────────────
        if "scan" in content.lower() or "sast" in content.lower():
            # SAST approved — deploy
            await self.message("orchestrator", "SAST approved. Pushing to GitHub and monitoring CI...", task_id)
            return "Deployment complete. CI passed. Model live."

        if "eda" in content.lower() or "feature" in content.lower():
            # Data Scientist sent EDA findings
            await self.report("EDA findings received. Updating pipeline with recommended feature transforms.", task_id)
            return None

        # Default: build pipeline
        await self.report("Loading dataset and running preprocessing checks...", task_id)
        await self.message("data_scientist", "Starting training. Please run EDA and flag any feature issues.", task_id)

        # Mock self-healing loop
        await self.report("Training complete. Accuracy: 94.1%. Sending code to SAST for review before deploy.", task_id)
        await self.message("sast", "ML pipeline training script ready. Please scan before I push to GitHub.", task_id)

        return None  # Will respond after SAST approval

    def get_tools(self):
        """Phase 2+: return list of LangChain tools."""
        return []  # TODO: code_execution, github_tools, mlflow_tools
