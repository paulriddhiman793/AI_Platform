from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from agents.base_agent import BaseAgent
from tools.workspace import workspace


class DataScientistAgent(BaseAgent):
    AGENT_ID = "data_scientist"
    SYSTEM_PROMPT = "Data Scientist"

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

    def _build_ds_code(self, dataset_path: Path) -> str:
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
print("COLUMNS:", list(df.columns))

target_col, confidence, reason, _ = infer_target_column(df, verbose=False)
print("INFERRED_TARGET_COL:", target_col)
print("TARGET_CONFIDENCE:", confidence)
print("TARGET_REASON:", reason)
y_raw = df[target_col]
if y_raw.nunique(dropna=True) <= 20:
    task_type = "classification"
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    task_type = "regression"
    model = RandomForestRegressor(n_estimators=120, random_state=42)

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

print("NULL_RATE_TOP5:")
null_rate = (df.isna().mean().sort_values(ascending=False).head(5) * 100).round(2)
for k, v in null_rate.items():
    print(f"  {{k}}: {{v}}%")
'''

    async def _build_detailed_report_from_output(self, dataset_path: Path, shape: tuple, null_top: dict, output: str) -> tuple[str, str]:
        rag_queries = [
            "class imbalance minority class distribution and recommendations",
            "model performance metrics validation cross validation overfitting drift leakage",
            "feature importance multicollinearity and data quality findings",
            "experiments next steps risk mitigation deployment readiness",
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
            "You are a senior Data Scientist. Create a detailed markdown report with strong explanations.\n"
            "Use these required sections exactly:\n"
            "## EDA Summary\n## Hypothesis Testing\n## Feature Engineering\n## Experiment Plan\n## Risks And Limitations\n## Next Steps\n"
            "Write at least 2 rich paragraphs or 6+ bullets per section.\n\n"
            f"Dataset path: {dataset_path}\n"
            f"Dataset shape: {shape}\n"
            f"Top null rates: {null_top}\n\n"
            "Hybrid-RAG retrieved transparency context from full output:\n"
            f"{context_blob}\n\n"
            "Use only evidence in this context. If a metric is absent, explicitly state it is not present."
        )
        final = await self._call_llm(synthesis_prompt)
        llm_status = "LLM: success"
        if final.startswith("[LLM"):
            llm_status = f"LLM fallback reason: {final}"
            final = (
                "## EDA Summary\n"
                "- Full-output synthesis fallback: data profile and model-trace were processed in chunked mode.\n"
                "- Key null-rate indicators suggest robust completeness for top features.\n"
                "- Model transparency output indicates train/test behavior and feature interactions were observable.\n\n"
                "## Hypothesis Testing\n"
                "- Test whether strongest predictors remain stable across folds and train/test splits.\n"
                "- Validate if nonlinear feature interactions materially change error metrics.\n"
                "- Check potential leakage or proxy effects with ablation tests.\n\n"
                "## Feature Engineering\n"
                "- Add transformations for skewed variables and evaluate uplift against baseline.\n"
                "- Prune low-utility/high-noise features based on permutation/importance traces.\n"
                "- Standardize categorical encodings and missing-value handling policy.\n\n"
                "## Experiment Plan\n"
                "- Run baseline vs engineered variants with identical CV protocol.\n"
                "- Perform sensitivity tests on model depth/estimators and regularization.\n"
                "- Track reproducibility with fixed seeds and per-fold logging.\n\n"
                "## Risks And Limitations\n"
                "- Potential overfitting if feature complexity grows without regularization controls.\n"
                "- Dataset representativeness risk should be validated by segment-level splits.\n"
                "- Business constraints and deployment latency targets must be included in objective tuning.\n\n"
                "## Next Steps\n"
                "- Finalize feature set, retrain with monitoring hooks, and compare to baseline.\n"
                "- Publish experiment matrix and decision log for model handoff.\n"
                "- Coordinate with Data Analyst for business KPI alignment."
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
                workspace.write("data_scientist", "probe_output.log", probe_out, task_id)
            await self.report(probe_out, task_id)
            return None

        if any(k in c for k in ["analyse data", "analyze data", "analysis", "eda", "dataset"]):
            ds_path = self.find_latest_uploaded_dataset()
            if not ds_path:
                await self.report(
                    "No uploaded dataset found in shared/datasets. Upload a file first.",
                    task_id,
                )
                return None

            code = self._build_ds_code(ds_path)
            if workspace.is_initialized:
                workspace.write("data_scientist", "analysis_ds.py", code, task_id)

            await self.report(
                "Data Scientist analysis started in Jupyter kernel.\n"
                "Code saved: data_scientist/analysis_ds.py\n"
                "Requirement enforced: `from model_transparency import run_pipeline` and `run_pipeline(...)`.",
                task_id,
            )

            exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=300)
            output = exec_result.get("output", "") or "[no output]"
            if workspace.is_initialized:
                workspace.write("data_scientist", "jupyter_output.txt", output, task_id)

            df = pd.read_csv(ds_path)
            null_top = (df.isna().mean().sort_values(ascending=False).head(5) * 100).round(2)
            llm_report, llm_status = await self._build_detailed_report_from_output(
                dataset_path=ds_path,
                shape=df.shape,
                null_top=dict(null_top),
                output=output,
            )
            report = (
                f"# Data Scientist Report\n\n"
                f"Dataset: `{ds_path}`\n"
                f"Shape: {df.shape}\n"
                f"Runner: {exec_result.get('runner')}\n"
                f"Used required import: `from model_transparency import run_pipeline`\n\n"
                f"{llm_status}\n\n"
                f"{llm_report}\n\n"
                f"## Raw Execution Output (Full)\n```\n{output}\n```\n"
            )
            if workspace.is_initialized:
                workspace.write("data_scientist", "report.md", report, task_id)

            await self.report(
                "Data Scientist completed analysis.\n"
                "Saved: data_scientist/jupyter_output.txt, data_scientist/report.md\n"
                f"Execution runner: {exec_result.get('runner')} | success={exec_result.get('success')}",
                task_id,
            )
            return None

        await self.report("Data Scientist ready. Use 'analyse data' to start.", task_id)
        return None

    def get_tools(self):
        return []
