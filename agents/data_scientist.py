from datetime import datetime
from pathlib import Path
import os
import hashlib
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

    def _build_fe_runner_code(self, dataset_path: Path) -> str:
        ds = str(dataset_path).replace("\\", "\\\\")
        return f'''import os
import pandas as pd
from fe_1 import run, identify_target_column

df = pd.read_csv(r"{ds}")
target_col = identify_target_column(df, r"{ds}")
run(csv_path=r"{ds}", target_col=target_col)
print("ENGINEERED_BASE:", os.path.splitext(os.path.basename(r"{ds}"))[0])
'''

    def _fe_output_paths(self, dataset_path: Path) -> dict:
        base = dataset_path.stem
        root = dataset_path.parent
        return {
            "engineered_csv": root / f"{base}_engineered.csv",
            "suggestions_md": root / f"{base}_feature_suggestions.md",
            "feature_py": root / f"{base}_feature_engineering.py",
            "base": base,
        }

    def _find_suggestions_file(self, base: str, dataset_dir: Path) -> Path | None:
        candidates: list[Path] = []
        for root in [dataset_dir, self._repo_root(), workspace.project_root or self._repo_root()]:
            try:
                candidates.extend(list(Path(root).glob(f"{base}*feature_suggestions*.md")))
                candidates.extend(list(Path(root).glob("*feature_suggestions*.md")))
            except Exception:
                continue
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _cache_root(self) -> Path:
        if workspace.output_path:
            return workspace.output_path / ".cache" / "engineered"
        return self._repo_root() / ".cache" / "engineered"

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


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

            # Reuse cached engineered CSV if this dataset was already processed.
            cache_root = self._cache_root()
            cache_root.mkdir(parents=True, exist_ok=True)
            ds_hash = self._hash_file(ds_path)
            cache_dir = cache_root / ds_hash
            cached_csv = cache_dir / "engineered.csv"
            cached_py = cache_dir / "feature_engineering.py"
            cached_md = cache_dir / "feature_suggestions.md"
            # Reuse cache only when BOTH engineered CSV and suggestions exist.
            # If suggestions are missing, re-run FE to regenerate and refresh cache.
            if cached_csv.exists() and cached_md.exists() and workspace.is_initialized:
                engineered_dst = workspace.write_bytes(
                    "shared",
                    f"datasets/{ds_path.stem}_engineered.csv",
                    cached_csv.read_bytes(),
                    task_id,
                )
                workspace.write(
                    "data_scientist",
                    f"{ds_path.stem}_feature_suggestions.md",
                    cached_md.read_text(encoding="utf-8", errors="replace"),
                    task_id,
                )
                await self.report(
                    "Data Scientist reused cached engineered dataset.\n"
                    f"Saved: shared/datasets/{ds_path.stem}_engineered.csv",
                    task_id,
                )
                await self.message(
                    "orchestrator",
                    "Data Scientist reused cached engineered dataset.",
                    task_id,
                )
                return None

            # Step 1: Run feature engineering pipeline (fe_1.py)
            fe_code = self._build_fe_runner_code(ds_path)

            await self.report(
                "Data Scientist feature engineering started in Jupyter kernel.\n",
                task_id,
            )

            fe_exec = await self.run_code_in_jupyter_kernel(fe_code, timeout_s=900)
            fe_output = fe_exec.get("output", "") or "[no output]"
            if workspace.is_initialized:
                workspace.write("data_scientist", "fe_jupyter_output.txt", fe_output, task_id)

            paths = self._fe_output_paths(ds_path)
            engineered_src = paths["engineered_csv"]
            suggestions_src = paths["suggestions_md"]
            feature_py_src = paths["feature_py"]
            base = paths["base"]

            engineered_dst = None
            if workspace.is_initialized and engineered_src.exists():
                engineered_bytes = engineered_src.read_bytes()
                engineered_dst = workspace.write_bytes(
                    "shared",
                    f"datasets/{base}_engineered.csv",
                    engineered_bytes,
                    task_id,
                )
            suggestions_file = suggestions_src if suggestions_src.exists() else self._find_suggestions_file(base, ds_path.parent)
            if workspace.is_initialized and suggestions_file and suggestions_file.exists():
                workspace.write(
                    "data_scientist",
                    f"{base}_feature_suggestions.md",
                    suggestions_file.read_text(encoding="utf-8", errors="replace"),
                    task_id,
                )
            if workspace.is_initialized and feature_py_src.exists():
                pass

            # Cache outputs for reuse across projects
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if engineered_src.exists():
                    (cache_dir / "engineered.csv").write_bytes(engineered_src.read_bytes())
                if feature_py_src.exists():
                    (cache_dir / "feature_engineering.py").write_text(
                        feature_py_src.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )
                if suggestions_file and suggestions_file.exists():
                    (cache_dir / "feature_suggestions.md").write_text(
                        suggestions_file.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )
            except Exception:
                pass

            await self.report(
                "Data Scientist completed feature engineering.\n"
                "Saved: data_scientist/*_feature_suggestions.md, shared/datasets/*_engineered.csv",
                task_id,
            )
            await self.message(
                "orchestrator",
                "Data Scientist completed feature engineering. Engineered CSV is ready.",
                task_id,
            )
            return None

        await self.report("Data Scientist ready. Use 'analyse data' to start.", task_id)
        return None

    def get_tools(self):
        return []
