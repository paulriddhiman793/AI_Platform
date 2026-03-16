from pathlib import Path
import re
import json
import io
#
import pandas as pd

from agents.base_agent import BaseAgent
from tools.workspace import workspace


class DataAnalystAgent(BaseAgent):
    AGENT_ID = "data_analyst"
    LLM_MODEL = "llama-3.1-8b-instant"
    SYSTEM_PROMPT = (
        "You are a senior Data Analyst focused on business impact. "
        "Be precise, structured, and evidence-driven. "
        "Translate technical findings into business implications, risks, and actions. "
        "Avoid speculation: if a metric is not present, state it explicitly. "
        "When asked for reports, use clear section headers and concise but complete explanations."
    )

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

    @staticmethod
    def _extract_output_meta(output: str) -> tuple[str | None, int | None]:
        target = None
        features = None
        if not output:
            return target, features
        for line in output.splitlines():
            if "TARGET_COL:" in line or "INFERRED_TARGET_COL:" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    target = parts[1].strip()
            if line.strip().lower().startswith("features:"):
                try:
                    features = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
        return target, features

    def _build_da_code(self, dataset_path: Path) -> str:
        tpl = Path("tools/da_pipeline_template.py").read_text(encoding="utf-8")
        ds = str(dataset_path).replace("\\", "\\\\")
        return tpl.replace("__DATA_PATH__", ds)

    def _find_latest_raw_dataset(self) -> Path | None:
        if not workspace.is_initialized or not workspace.project_root:
            return None
        ds_dir = workspace.project_root / "shared" / "datasets"
        if not ds_dir.exists():
            return None
        candidates = []
        for ext in ("*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json", "*.tsv"):
            candidates.extend(ds_dir.glob(ext))
        if not candidates:
            return None
        raw = [p for p in candidates if "engineered" not in p.name.lower()]
        if raw:
            return sorted(raw, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    @staticmethod
    def _extract_results_block(output: str) -> dict | None:
        if not output:
            return None
        begin = "__DA_RESULTS_BEGIN__"
        end = "__DA_RESULTS_END__"
        if begin not in output or end not in output:
            return None
        blob = output.split(begin, 1)[1].split(end, 1)[0].strip()
        if not blob:
            return None
        try:
            return json.loads(blob)
        except Exception:
            return None

    @staticmethod
    def _build_findings_txt(payload: dict) -> str:
        if not payload:
            return "No structured results were produced."
        ov = payload.get("overview", {})
        dq = payload.get("data_quality", {})
        miss = payload.get("missing_value_analysis", {})
        dup = payload.get("duplicate_analysis", {})
        out = payload.get("outlier_analysis", {})
        corr = payload.get("correlation_analysis", {})
        chi = payload.get("chi_square_tests", [])
        anomalies = payload.get("anomaly_flags", [])
        string_quality = payload.get("string_quality", {})
        top_outliers = sorted(
            [(k, v.get("iqr_outliers", 0)) for k, v in out.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        top_corr = corr.get("top_pairs", [])[:5] if isinstance(corr, dict) else []
        miss_cols = []
        per_col = miss.get("per_column", {})
        if isinstance(per_col, dict):
            miss_cols = sorted(
                [(k, v.get("pct", 0)) for k, v in per_col.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        sig_chi = [t for t in chi if t.get("significant")] if isinstance(chi, list) else []
        lines = [
            "DATA ANALYST FINDINGS",
            "=" * 40,
            f"Rows: {ov.get('n_rows')} | Columns: {ov.get('n_cols')}",
            f"Numeric cols: {len(ov.get('numeric_cols', []))} | Categorical cols: {len(ov.get('categorical_cols', []))}",
            f"Missing cells: {miss.get('total_missing_pct', 0)}% | Columns affected: {miss.get('cols_with_missing', 0)}",
            f"Duplicate rows: {dup.get('duplicate_rows', 0)} ({dup.get('duplicate_rows_pct', 0)}%)",
            f"Data quality score: {dq.get('score')} ({dq.get('label')})",
            "",
            "Top outlier columns (IQR count):",
        ]
        if top_outliers:
            for name, cnt in top_outliers:
                lines.append(f"- {name}: {cnt}")
        else:
            lines.append("- None detected")
        lines.append("")
        lines.append("Missingness by column (top 5):")
        if miss_cols:
            for name, pct in miss_cols:
                lines.append(f"- {name}: {pct}%")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("Top correlations (Pearson):")
        if top_corr:
            for p in top_corr:
                lines.append(
                    f"- {p.get('col1')} vs {p.get('col2')}: r={p.get('pearson_r')} ({p.get('strength')})"
                )
        else:
            lines.append("- None")
        lines.append("")
        lines.append("Significant categorical associations (Chi-square):")
        if sig_chi:
            for t in sig_chi[:5]:
                lines.append(
                    f"- {t.get('col1')} vs {t.get('col2')}: CramerV={t.get('cramers_v')} ({t.get('association')})"
                )
        else:
            lines.append("- None")
        lines.append("")
        lines.append(f"Anomalies flagged: {len(anomalies)}")
        if string_quality:
            lines.append(f"String quality issues: {len(string_quality)} columns")
        lines.append("")
        lines.append("FULL PAYLOAD (TEXT)")
        lines.append("=" * 40)
        lines.extend(DataAnalystAgent._render_payload_text(payload))
        return "\n".join(lines)

    @staticmethod
    def _render_payload_text(payload: dict) -> list[str]:
        def _fmt_value(v):
            if isinstance(v, float):
                return f"{v:.6f}".rstrip("0").rstrip(".")
            return str(v)

        def _render_dict(d: dict, prefix: str = "") -> list[str]:
            out = []
            for key in sorted(d.keys()):
                val = d[key]
                if isinstance(val, dict):
                    out.append(f"{prefix}{key}:")
                    out.extend(_render_dict(val, prefix + "  "))
                elif isinstance(val, list):
                    out.append(f"{prefix}{key}:")
                    out.extend(_render_list(val, prefix + "  "))
                else:
                    out.append(f"{prefix}{key}: {_fmt_value(val)}")
            return out

        def _render_list(lst: list, prefix: str = "") -> list[str]:
            out = []
            if not lst:
                out.append(f"{prefix}(empty)")
                return out
            # simple list
            if all(not isinstance(x, (dict, list)) for x in lst):
                out.append(f"{prefix}{', '.join(_fmt_value(x) for x in lst)}")
                return out
            # list of dicts / lists
            for i, item in enumerate(lst, 1):
                if isinstance(item, dict):
                    out.append(f"{prefix}item {i}:")
                    out.extend(_render_dict(item, prefix + "  "))
                elif isinstance(item, list):
                    out.append(f"{prefix}item {i}:")
                    out.extend(_render_list(item, prefix + "  "))
                else:
                    out.append(f"{prefix}item {i}: {_fmt_value(item)}")
            return out

        # Preserve a stable top-level order
        section_intros = {
            "duplicate_analysis": (
                "Purpose: detects exact duplicate rows that can inflate metrics or skew model training. "
                "How to read: focus on duplicate_rows and duplicate_rows_pct; any non-zero value means you should deduplicate before modeling."
            ),
            "dtype_audit": (
                "Purpose: checks whether columns are stored in the wrong data type (e.g., numbers as text or booleans as ints). "
                "How to read: each issue lists the column and the suspected type problem; fix these to avoid parsing and model errors."
            ),
            "date_analysis": (
                "Purpose: validates date-like columns and their ranges, including future dates and parse failures. "
                "How to read: review min/max/span, n_failed, and has_future_dates to judge date quality and coverage."
            ),
            "cardinality_audit": (
                "Purpose: measures uniqueness to flag ID-like columns, constants, or high-card categoricals. "
                "How to read: check the tag (constant, binary, id_like, low_card, high_card) and pct_unique to guide encoding choices."
            ),
            "anomaly_flags": (
                "Purpose: surfaces suspicious values such as negatives where not expected, zeros in critical fields, mixed types, or future dates. "
                "How to read: each entry gives the anomaly type and count; investigate high counts for data quality issues."
            ),
            "string_quality": (
                "Purpose: audits categorical text for hidden quality problems like extra whitespace, casing inconsistencies, or empty strings. "
                "How to read: issue counts show where category cleanup will improve grouping and modeling."
            ),
            "correlation_analysis": (
                "Purpose: quantifies linear relationships between numeric features. "
                "How to read: top_pairs highlight strong associations; large absolute r indicates potentially redundant features or strong drivers."
            ),
            "chi_square_tests": (
                "Purpose: tests statistical association between categorical variables. "
                "How to read: significant p-values plus higher Cramer's V indicate meaningful relationships between categories."
            ),
            "categorical_frequencies": (
                "Purpose: shows the distribution of categories for each categorical column. "
                "How to read: top_categories and mode_pct reveal dominant groups and possible imbalance."
            ),
            "time_series_trends": (
                "Purpose: summarizes month-level trends when date columns exist. "
                "How to read: monthly aggregates show changes over time and seasonality for key numeric fields."
            ),
            "hypothesis_tests": (
                "Purpose: tests whether numeric outcomes differ across categorical groups (binary and multi-group). "
                "How to read: significant p-values and effect sizes indicate real group differences worth business attention."
            ),
            "segment_profiles": (
                "Purpose: profiles numeric behavior by category to explain how segments differ. "
                "How to read: compare per-segment means/medians to identify high-value or at-risk segments."
            ),
            "kpi_summary": (
                "Purpose: aggregates key numeric fields into totals, averages, and non-zero rates. "
                "How to read: use this as a quick business snapshot and to validate KPI magnitudes."
            ),
            "cohort_analysis": (
                "Purpose: tracks KPI aggregates over time cohorts when dates exist. "
                "How to read: per-period summaries show growth, decline, or cohort stability."
            ),
        }
        ordered_keys = [
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
        lines = []
        for key in ordered_keys:
            if key not in payload:
                lines.append(f"{key}: [missing]")
                continue
            val = payload[key]
            lines.append(f"{key}:")
            if key in section_intros:
                lines.append(f"  intro: {section_intros[key]}")
            if isinstance(val, dict):
                if key == "cohort_analysis" and not val:
                    lines.append("  note: empty (no usable datetime columns or insufficient data)")
                else:
                    lines.extend(_render_dict(val, "  "))
            elif isinstance(val, list):
                lines.extend(_render_list(val, "  "))
            else:
                lines.append(f"  {_fmt_value(val)}")
            lines.append("")
        return lines[:-1] if lines and lines[-1] == "" else lines

    def _write_graph_bytes(self, fig, filename: str, task_id: str | None) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        if workspace.is_initialized:
            workspace.write_bytes("data_analyst", filename, buf.read(), task_id)

    def _write_da_graphs(self, dataset_path: Path, payload: dict, task_id: str | None) -> list[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return []

        df = pd.read_csv(dataset_path)
        numeric_cols = payload.get("overview", {}).get("numeric_cols", []) if payload else []
        cat_cols = payload.get("overview", {}).get("categorical_cols", []) if payload else []
        saved = []

        miss = payload.get("missing_value_analysis", {}).get("per_column", {}) if payload else {}
        if miss:
            cols = list(miss.keys())
            pcts = [miss[c]["pct"] for c in cols]
            fig = plt.figure(figsize=(8, 3.5))
            plt.bar(cols, pcts, color="#3b82f6")
            plt.title("Missingness (%) by Column")
            plt.xticks(rotation=60, ha="right", fontsize=8)
            plt.tight_layout()
            name = "graphs/missingness.png"
            self._write_graph_bytes(fig, name, task_id)
            plt.close(fig)
            saved.append(name)

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(method="pearson")
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("Pearson Correlation (Numeric)")
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=6)
            plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=6)
            plt.tight_layout()
            name = "graphs/correlation.png"
            self._write_graph_bytes(fig, name, task_id)
            plt.close(fig)
            saved.append(name)

        for col in numeric_cols[:6]:
            fig = plt.figure(figsize=(5, 3.2))
            plt.hist(df[col].dropna(), bins=30, color="#10b981", alpha=0.85)
            plt.title(f"Distribution: {col}")
            plt.tight_layout()
            name = f"graphs/hist_{col}.png"
            self._write_graph_bytes(fig, name, task_id)
            plt.close(fig)
            saved.append(name)

        for col in cat_cols[:4]:
            vc = df[col].value_counts(dropna=False).head(10)
            fig = plt.figure(figsize=(6, 3.2))
            plt.bar(vc.index.astype(str), vc.values, color="#f59e0b")
            plt.title(f"Top Categories: {col}")
            plt.xticks(rotation=60, ha="right", fontsize=8)
            plt.tight_layout()
            name = f"graphs/cat_{col}.png"
            self._write_graph_bytes(fig, name, task_id)
            plt.close(fig)
            saved.append(name)

        return saved

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

        target_col, feature_count = self._extract_output_meta(output)
        features_text = str(feature_count) if feature_count is not None else "unknown"
        target_text = target_col or "unknown"
        synthesis_prompt = (
            "You are a senior Data Analyst. Create a concise business report.\n"
            "Use these sections exactly:\n"
            "## Executive Summary\n## Business Insights\n"
            "Write 6+ bullets per section.\n"
            "Business Insights must be very informative and eye-catching (clear, high-signal phrasing).\n"
            "Completion is mandatory: include both sections and finish with END_OF_REPORT.\n"
            "Do NOT use tables. Use bullet points only.\n\n"
            f"Dataset path: {dataset_path}\n"
            f"Shape: {shape}\n"
            f"Overall null %: {null_pct:.2f}\n"
            f"From output.txt: FEATURES={features_text}, TARGET_COL={target_text}\n\n"
            "Hybrid-RAG retrieved transparency context from full output:\n"
            f"{context_blob}\n\n"
            "Use only evidence in this context. If a metric is absent, explicitly state it is not present.\n"
            "End your response with the line: END_OF_REPORT"
        )
        final = await self._call_llm(synthesis_prompt, model_override=self.LLM_MODEL)
        if "END_OF_REPORT" not in final:
            # Retry once with a shorter context to reduce truncation risk.
            short_ctx = self._extract_class_imbalance_block(output)
            retry_prompt = synthesis_prompt.replace(context_blob, short_ctx)
            final = await self._call_llm(retry_prompt, model_override=self.LLM_MODEL)
        llm_status = "LLM: success"
        if final.startswith("[LLM"):
            llm_status = "Data analyst is busy. Please try again later."
            final = ""
        final = final.replace("END_OF_REPORT", "").strip()
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
            ds_path = self._find_latest_raw_dataset()
            if not ds_path:
                await self.report(
                    "No uploaded dataset found in shared/datasets. Upload a file first.",
                    task_id,
                )
                return None

            code = self._build_da_code(ds_path)

            await self.report(
                "Data Analyst analysis started in Jupyter kernel.\n"
                "Pipeline: full automated EDA (da_pipeline).",
                task_id,
            )

            exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=300)
            output = exec_result.get("output", "") or "[no output]"
            if workspace.is_initialized:
                workspace.write("data_analyst", "jupyter_output.txt", output, task_id)

            payload = self._extract_results_block(output)
            if workspace.is_initialized and payload:
                workspace.write("data_analyst", "da_results.json", json.dumps(payload, indent=2), task_id)

            findings = self._build_findings_txt(payload)
            if workspace.is_initialized:
                workspace.write("data_analyst", "findings.txt", findings, task_id)
                if payload:
                    workspace.write("data_analyst", "da_result.txt", json.dumps(payload, indent=2), task_id)

            graphs = self._write_da_graphs(ds_path, payload or {}, task_id)

            await self.report(
                "Data Analyst completed analysis.\n"
                "Saved: data_analyst/jupyter_output.txt, data_analyst/da_results.json, data_analyst/da_result.txt, data_analyst/findings.txt\n"
                + (f"Graphs: {', '.join(graphs)}\n" if graphs else "Graphs: none\n")
                + f"Execution runner: {exec_result.get('runner')} | success={exec_result.get('success')}",
                task_id,
            )
            await self.message(
                "orchestrator",
                "Data Analyst completed automated EDA. Results + graphs are ready.",
                task_id,
            )
            return None

        await self.report("Data Analyst ready. Use 'analyse data' to start.", task_id)
        return None

    def get_tools(self):
        return []
