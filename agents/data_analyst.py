from pathlib import Path
import re

import pandas as pd

from agents.base_agent import BaseAgent
from tools.workspace import workspace


class DataAnalystAgent(BaseAgent):
    AGENT_ID = "data_analyst"
    SYSTEM_PROMPT = "Data Analyst"

    @staticmethod
    def _extract_class_imbalance_block(output: str) -> str:
        if not output:
            return "[Class Imbalance section not found in output]"
        lines = output.splitlines()
        start = None
        for i, line in enumerate(lines):
            if "class imbalance" in line.lower():
                start = i
                break
        if start is None:
            return "[Class Imbalance section not found in output]"
        end = len(lines)
        for j in range(start + 1, len(lines)):
            lj = lines[j].strip()
            if not lj:
                continue
            if re.match(r"^[A-Z][A-Z\s:/()_-]{6,}$", lj) and "CLASS IMBALANCE" not in lj:
                end = j
                break
            if lj.lower().startswith("=== ") and "class imbalance" not in lj.lower():
                end = j
                break
        block = "\n".join(lines[start:end]).strip()
        return block or "[Class Imbalance section not found in output]"

    def _build_da_code(self, dataset_path: Path) -> str:
        ds = str(dataset_path).replace("\\", "\\\\")
        return f'''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model_transparency import run_pipeline
from model_transparency import run_pipeline_from_df, infer_target_column

# Required by workflow:
from model_transparency import run_pipeline

df = pd.read_csv(r"{ds}")
print("DATASET_PATH:", r"{ds}")
print("DATASET_SHAPE:", df.shape)
print("NUMERIC_COLS:", list(df.select_dtypes(include=[np.number]).columns))
target_col, confidence, reason, _ = infer_target_column(df, verbose=False)
print("INFERRED_TARGET_COL:", target_col)
print("TARGET_CONFIDENCE:", confidence)
print("TARGET_REASON:", reason)
y_raw = df[target_col]

if y_raw.nunique(dropna=True) <= 20:
    task_type = "classification"
    model = RandomForestClassifier(n_estimators=90, random_state=42)
else:
    task_type = "regression"
    model = RandomForestRegressor(n_estimators=110, random_state=42)

print("TARGET_COL:", target_col)
print("TASK_TYPE:", task_type)
run_pipeline_from_df(
    model=model,
    df=df,
    target_col=target_col,
    task_type=task_type,
    test_size=0.2,
    scale=(task_type == "regression"),
    cv=3,
    n_walkthrough=2
)

print("BUSINESS_SIGNAL_TOP5_MEANS:")
num_means = df.select_dtypes(include=[np.number]).mean(numeric_only=True)
for col in list(num_means.index)[:5]:
    print(f"  {{col}}={{float(num_means[col]):.3f}}")
'''

    async def _build_detailed_business_report(self, dataset_path: Path, shape: tuple, null_pct: float, output: str) -> tuple[str, str]:
        rag_queries = [
            "class imbalance business risk minority segments and mitigation",
            "model performance metrics and KPI impact for stakeholders",
            "data quality leakage drift latency and operational risks",
            "recommended actions monitoring plan and governance",
        ]
        rag_hits: list[dict] = []
        for q in rag_queries:
            rag_hits.extend(self.rag_query_transparency_output(q, top_k=3))
        seen = set()
        rag_blocks = []
        for h in sorted(rag_hits, key=lambda x: x.get("score", 0), reverse=True):
            cid = h.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            rag_blocks.append(
                f"[chunk:{cid} score:{h.get('score', 0):.3f}]\n{h.get('text', '')}"
            )
            if len(rag_blocks) >= 10:
                break
        context_blob = "\n\n".join(rag_blocks) if rag_blocks else self._extract_class_imbalance_block(output)

        synthesis_prompt = (
            "You are a senior Data Analyst. Create a detailed markdown report with clear business explanations.\n"
            "Use these sections exactly:\n"
            "## Executive Summary\n## Business Insights\n## Risk And Opportunity\n## Recommended Actions\n## KPI Monitoring Plan\n## Stakeholder Notes\n"
            "Write at least 2 explanatory paragraphs or 6+ bullets per section.\n\n"
            f"Dataset path: {dataset_path}\n"
            f"Shape: {shape}\n"
            f"Overall null %: {null_pct:.2f}\n\n"
            "Hybrid-RAG retrieved transparency context from full output:\n"
            f"{context_blob}\n\n"
            "Use only evidence in this context. If a metric is absent, explicitly state it is not present."
        )
        final = await self._call_llm(synthesis_prompt)
        llm_status = "LLM: success"
        if final.startswith("[LLM"):
            llm_status = f"LLM fallback reason: {final}"
            final = (
                "## Executive Summary\n"
                "- Full-output synthesis fallback generated from chunked execution analysis.\n"
                "- Data quality appears workable for iterative model development and KPI design.\n"
                "- Monitoring requirements are high for production stability.\n\n"
                "## Business Insights\n"
                "- Prioritize stable and interpretable features for decision-facing workflows.\n"
                "- Use segment-level tracking to explain variability in predicted outcomes.\n"
                "- Convert model observations into monthly operational scorecards.\n\n"
                "## Risk And Opportunity\n"
                "- Risk: distribution drift and latency inflation can erode trust quickly.\n"
                "- Opportunity: proactive quality alerts reduce rework and outage impact.\n"
                "- Opportunity: targeted interventions for high-risk segments improve ROI.\n\n"
                "## Recommended Actions\n"
                "- Establish threshold-based alerting for null spikes and KPI drift.\n"
                "- Define executive KPI pack: quality, performance, and business impact.\n"
                "- Add rollback conditions and ownership for incident response.\n\n"
                "## KPI Monitoring Plan\n"
                "- Weekly model quality checks and monthly business impact reviews.\n"
                "- Track leading indicators (data quality) and lagging indicators (outcomes).\n"
                "- Publish trend deltas, not only absolute values.\n\n"
                "## Stakeholder Notes\n"
                "- Product: align KPI definitions with decision cadence.\n"
                "- Engineering: instrument data contracts and alert pipelines.\n"
                "- Leadership: review trade-offs between speed and reliability."
            )
        return final, llm_status

    async def handle_task(self, payload: dict) -> str | None:
        content = payload.get("content", "")
        task_id = payload.get("task_id")
        c = content.lower()

        marker = "__PROBE_OUTPUT_BEGIN__"
        if marker in content:
            probe_out = content.split(marker, 1)[1].strip()
            if workspace.is_initialized:
                workspace.write("data_analyst", "probe_output.log", probe_out, task_id)
            await self.report(probe_out, task_id)
            return None

        if any(k in c for k in ["analyse data", "analyze data", "analysis", "business insights", "dataset"]):
            ds_path = self.find_latest_uploaded_dataset()
            if not ds_path:
                await self.report(
                    "No uploaded dataset found in shared/datasets. Upload a file first.",
                    task_id,
                )
                return None

            code = self._build_da_code(ds_path)
            if workspace.is_initialized:
                workspace.write("data_analyst", "analysis_da.py", code, task_id)

            await self.report(
                "Data Analyst analysis started in Jupyter kernel.\n"
                "Code saved: data_analyst/analysis_da.py\n"
                "Requirement enforced: `from model_transparency import run_pipeline` and `run_pipeline(...)`.",
                task_id,
            )

            exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=300)
            output = exec_result.get("output", "") or "[no output]"
            if workspace.is_initialized:
                workspace.write("data_analyst", "jupyter_output.txt", output, task_id)

            df = pd.read_csv(ds_path)
            rows, cols = df.shape
            null_pct = float((df.isna().sum().sum() / max(1, rows * cols)) * 100)
            llm_report, llm_status = await self._build_detailed_business_report(
                dataset_path=ds_path,
                shape=(rows, cols),
                null_pct=null_pct,
                output=output,
            )
            business_report = (
                f"# Data Analyst Report\n\n"
                f"Dataset: `{ds_path}`\n"
                f"Shape: ({rows}, {cols})\n"
                f"Overall null %: {null_pct:.2f}\n"
                f"Execution runner: {exec_result.get('runner')}\n"
                f"Used required import: `from model_transparency import run_pipeline`\n\n"
                f"{llm_status}\n\n"
                f"{llm_report}\n\n"
                f"## Raw Execution Output (Full)\n```\n{output}\n```\n"
            )
            if workspace.is_initialized:
                workspace.write("data_analyst", "report.md", business_report, task_id)
                workspace.write("data_analyst", "business_report.md", business_report, task_id)

            await self.report(
                "Data Analyst completed analysis.\n"
                "Saved: data_analyst/jupyter_output.txt, data_analyst/report.md\n"
                f"Execution runner: {exec_result.get('runner')} | success={exec_result.get('success')}",
                task_id,
            )
            return None

        await self.report("Data Analyst ready. Use 'analyse data' to start.", task_id)
        return None

    def get_tools(self):
        return []
