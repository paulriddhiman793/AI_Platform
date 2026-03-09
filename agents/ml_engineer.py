from datetime import datetime

from agents.base_agent import BaseAgent
from tools.workspace import workspace


class MLEngineerAgent(BaseAgent):
    AGENT_ID = "ml_engineer"
    SYSTEM_PROMPT = "ML Engineer"

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        c = content.lower()

        marker = "__PROBE_OUTPUT_BEGIN__"
        if marker in content:
            probe_out = content.split(marker, 1)[1].strip()
            if workspace.is_initialized:
                workspace.write("ml_engineer", "probe_output.log", probe_out, task_id)
            await self.report(probe_out, task_id)
            return None

        if any(k in c for k in ["model", "train", "eda", "dataset", "pipeline", "predict", "classification", "deploy"]):
            await self.report("PROBE_RUN_START [ml_engineer] Running transparency_quick_probe.py", task_id)
            result = await self.run_model_transparency_probe(task_id=task_id)
            run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            out = result.get("output", "") or "[no output]"
            log = (
                f"agent=ml_engineer\n"
                f"timestamp={run_ts}\n"
                f"success={result.get('success')}\n"
                f"script={result.get('script')}\n\n"
                f"{out}\n"
            )
            if workspace.is_initialized:
                workspace.write("ml_engineer", "probe_output.log", log, task_id)

            await self.report(
                "PROBE_RUN_END [ml_engineer]\n"
                f"success={result.get('success')} | script={result.get('script')}\n"
                "Saved: ml_engineer/probe_output.log\n"
                f"Output:\n{out}",
                task_id,
            )
            return None

        await self.report("ML Engineer ready. Waiting for model-related instruction.", task_id)
        return None

    def get_tools(self):
        return []
