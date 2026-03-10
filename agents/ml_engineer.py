from datetime import datetime
import os
import time
import asyncio
import ast
import re
from pathlib import Path
import os
from agents.base_agent import BaseAgent
from tools.workspace import workspace


class MLEngineerAgent(BaseAgent):
    AGENT_ID = "ml_engineer"
    SYSTEM_PROMPT = "ML Engineer"
    _GEMINI_CLIENT = None
    _GEMINI_LOCK: asyncio.Lock | None = None
    _GEMINI_SEMAPHORE: asyncio.Semaphore | None = None
    _GEMINI_REQ_TIMES: list[float] = []
    _GEMINI_TOKEN_WINDOW: list[tuple[float, int]] = []
    _GEMINI_DAY: str | None = None
    _GEMINI_DAY_COUNT: int = 0

    @staticmethod
    def _extract_code(text: str) -> str:
        if not text:
            return ""
        fence = "```"
        if fence in text:
            parts = text.split(fence)
            # Fenced blocks live at odd indexes: 1,3,5...
            for block in parts[1::2]:
                blk = block.strip()
                if blk.startswith("python"):
                    return blk.split("\n", 1)[1].strip() if "\n" in blk else ""
                if blk.startswith("py"):
                    return blk.split("\n", 1)[1].strip() if "\n" in blk else ""
            # Fallback to first fenced block
            if len(parts) > 1:
                blk = parts[1].strip()
                head = blk.split("\n", 1)[0].strip().lower()
                if head in ("python", "py") and "\n" in blk:
                    return blk.split("\n", 1)[1].strip()
                return blk
            return ""
        return text.strip()

    @classmethod
    def _gemini_lock(cls) -> asyncio.Lock:
        if cls._GEMINI_LOCK is None:
            cls._GEMINI_LOCK = asyncio.Lock()
        return cls._GEMINI_LOCK

    @classmethod
    def _gemini_semaphore(cls) -> asyncio.Semaphore:
        if cls._GEMINI_SEMAPHORE is None:
            raw = (os.getenv("GEMINI_MAX_CONCURRENCY") or "1").strip()
            try:
                val = max(1, min(4, int(raw)))
            except Exception:
                val = 1
            cls._GEMINI_SEMAPHORE = asyncio.Semaphore(val)
        return cls._GEMINI_SEMAPHORE

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough estimate: 4 chars per token
        return max(1, int(len(text) / 4))

    async def _gemini_rate_limit(self, tokens: int) -> str | None:
        rpm = 30
        tpm = 15000
        rpd = 14400
        now = time.monotonic()
        day = datetime.utcnow().strftime("%Y-%m-%d")

        async with self._gemini_lock():
            if self._GEMINI_DAY != day:
                self._GEMINI_DAY = day
                self._GEMINI_DAY_COUNT = 0

            # RPD check
            if self._GEMINI_DAY_COUNT >= rpd:
                return "RPD limit reached"

            # Purge 60s windows
            self._GEMINI_REQ_TIMES = [t for t in self._GEMINI_REQ_TIMES if now - t < 60]
            self._GEMINI_TOKEN_WINDOW = [(t, tok) for (t, tok) in self._GEMINI_TOKEN_WINDOW if now - t < 60]

            # RPM wait
            if len(self._GEMINI_REQ_TIMES) >= rpm:
                wait = 60 - (now - self._GEMINI_REQ_TIMES[0])
                if wait > 0:
                    await asyncio.sleep(wait)

            # TPM wait
            token_sum = sum(tok for _, tok in self._GEMINI_TOKEN_WINDOW)
            if token_sum + tokens > tpm and self._GEMINI_TOKEN_WINDOW:
                oldest = self._GEMINI_TOKEN_WINDOW[0][0]
                wait = 60 - (now - oldest)
                if wait > 0:
                    await asyncio.sleep(wait)

            # Record usage
            now2 = time.monotonic()
            self._GEMINI_REQ_TIMES.append(now2)
            self._GEMINI_TOKEN_WINDOW.append((now2, tokens))
            self._GEMINI_DAY_COUNT += 1

        return None

    def _get_gemini_client(self):
        if self._GEMINI_CLIENT is not None:
            return self._GEMINI_CLIENT
        try:
            from google import genai
        except Exception:
            # Try lazy install if not present
            try:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"], check=False)
                from google import genai
            except Exception:
                return None
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            return None
        self._GEMINI_CLIENT = genai.Client(api_key=api_key)
        return self._GEMINI_CLIENT

    async def _call_llm(self, user_message: str, system_prompt: str = None) -> str:
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            await asyncio.sleep(0.05)
            return f"[LLM unavailable] {self.AGENT_ID}: GEMINI_API_KEY missing"

        client = self._get_gemini_client()
        if client is None:
            return f"[LLM unavailable] {self.AGENT_ID}: google-genai not installed"

        model = (os.getenv("GEMINI_MODEL") or "gemma-3-27b-it").strip()
        prompt = (system_prompt or self.SYSTEM_PROMPT) + "\n\n" + user_message
        tokens = self._estimate_tokens(prompt)

        rate_err = await self._gemini_rate_limit(tokens)
        if rate_err:
            return f"[LLM error] {rate_err}"

        def _sync_call() -> str:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            if not text:
                return "[LLM empty] no text returned"
            return text

        try:
            async with self._gemini_semaphore():
                return await asyncio.to_thread(_sync_call)
        except Exception as exc:
            return f"[LLM error] {type(exc).__name__}: {exc}"

    @staticmethod
    def _validate_python_code(code: str) -> tuple[bool, str]:
        if not code.strip():
            return False, "empty code"
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _ensure_import(code: str, token: str, import_stmt: str) -> tuple[str, bool]:
        if token not in code:
            return code, False
        if import_stmt in code:
            return code, False
        lines = code.splitlines()
        insert_idx = 0
        while insert_idx < len(lines):
            line = lines[insert_idx].strip()
            if not line or line.startswith("#") or line.startswith("import ") or line.startswith("from "):
                insert_idx += 1
                continue
            break
        lines.insert(insert_idx, import_stmt)
        return "\n".join(lines), True

    @classmethod
    def _sanitize_training_code(cls, code: str) -> tuple[str, str]:
        changed = []
        # Fix infer_target_column usage if assigned to single var
        if "infer_target_column" in code and "target_col" in code:
            if re.search(r"\btarget_col\s*=\s*infer_target_column\(", code):
                code = re.sub(
                    r"\btarget_col\s*=\s*infer_target_column\(([^)]*)\)",
                    r"target_col, _, _, _ = infer_target_column(\1, verbose=False)",
                    code,
                )
                changed.append("infer_target_column tuple-unpack")

        imports = [
            ("OneHotEncoder", "from sklearn.preprocessing import OneHotEncoder"),
            ("StandardScaler", "from sklearn.preprocessing import StandardScaler"),
            ("SimpleImputer", "from sklearn.impute import SimpleImputer"),
            ("ColumnTransformer", "from sklearn.compose import ColumnTransformer"),
            ("Pipeline", "from sklearn.pipeline import Pipeline"),
            ("train_test_split", "from sklearn.model_selection import train_test_split"),
            ("RandomForestClassifier", "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"),
            ("mean_squared_error", "from sklearn.metrics import mean_squared_error"),
            ("accuracy_score", "from sklearn.metrics import accuracy_score"),
            ("f1_score", "from sklearn.metrics import f1_score"),
            ("r2_score", "from sklearn.metrics import r2_score"),
        ]
        for token, stmt in imports:
            code, did = cls._ensure_import(code, token, stmt)
            if did:
                changed.append(f"import:{token}")

        return code, (", ".join(changed) if changed else "none")

    def _collect_rag_context(self) -> str:
        queries_output = [
            "class imbalance minority class distribution recommendations",
            "missing values leakage multicollinearity data quality",
            "feature engineering recommendations log transform drop",
            "model performance metrics overfitting validation",
        ]
        queries_reports = [
            "data scientist report feature engineering hypothesis testing",
            "data analyst report business insights risks actions",
        ]

        hits = []
        for q in queries_output:
            hits.extend(self.rag_query_transparency_output(q, top_k=3))
        for q in queries_reports:
            hits.extend(self.rag_query_reports(q, top_k=3))

        seen = set()
        blocks = []
        for h in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
            hid = h.get("id")
            if (hid, h.get("text", "")) in seen:
                continue
            seen.add((hid, h.get("text", "")))
            blocks.append(f"[chunk:{hid} score:{h.get('score', 0):.3f}]\n{h.get('text', '')}")
            if len(blocks) >= 12:
                break
        return "\n\n".join(blocks).strip()

    def _build_training_code(self, dataset_path: Path, rag_context: str) -> str:
        ds = str(dataset_path).replace("\\", "\\\\")
        imbalance_flag = "class imbalance" in rag_context.lower() or "imbalance" in rag_context.lower()
        class_weight_line = "        class_weight='balanced'," if imbalance_flag else ""
        return f'''import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from model_transparency import infer_target_column

df = pd.read_csv(r"{ds}")
target_col, confidence, reason, _ = infer_target_column(df, verbose=False)
print("DATASET_PATH:", r"{ds}")
print("DATASET_SHAPE:", df.shape)
print("TARGET_COL:", target_col)
print("TARGET_CONFIDENCE:", confidence)
print("TARGET_REASON:", reason)

y = df[target_col]
X = df.drop(columns=[target_col])

# Basic feature engineering: log1p on highly skewed numeric features.
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    try:
        skew = X[col].dropna().skew()
        if skew is not None and skew > 1:
            X[col] = np.log1p(X[col].clip(lower=0))
            print("LOG1P:", col, "skew=", round(float(skew), 3))
    except Exception:
        pass

cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
)

task_type = "classification" if y.nunique(dropna=True) <= 20 else "regression"
print("TASK_TYPE:", task_type)

if task_type == "classification":
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        {class_weight_line}
        n_jobs=-1,
    )
else:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if task_type == "classification" else None
)

pipe = Pipeline(steps=[("pre", pre), ("model", model)])
pipe.fit(X_train, y_train)

metrics = {{}}
if task_type == "classification":
    preds = pipe.predict(X_test)
    metrics["accuracy"] = float(accuracy_score(y_test, preds))
    metrics["f1"] = float(f1_score(y_test, preds, average="weighted"))
else:
    preds = pipe.predict(X_test)
    metrics["r2"] = float(r2_score(y_test, preds))
    metrics["rmse"] = float(mean_squared_error(y_test, preds) ** 0.5)

print("METRICS:", json.dumps(metrics, indent=2))

root = Path(r"{str(self._repo_root())}")
model_dir = root / "ml_engineer" / "model"
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_dir / "model.joblib")

shared = root / "shared"
shared.mkdir(parents=True, exist_ok=True)
(shared / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(root / "ml_engineer" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
'''

    def _build_feature_engineering_code(self, dataset_path: Path) -> str:
        ds = str(dataset_path).replace("\\", "\\\\")
        return f'''import numpy as np
import pandas as pd
from model_transparency import infer_target_column

df = pd.read_csv(r"{ds}")
target_col, confidence, reason, _ = infer_target_column(df, verbose=False)
print("DATASET_PATH:", r"{ds}")
print("DATASET_SHAPE:", df.shape)
print("TARGET_COL:", target_col)
print("TARGET_CONFIDENCE:", confidence)
print("TARGET_REASON:", reason)

y = df[target_col]
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Basic FE: log1p skewed numeric features, median/mode imputations.
skewed = []
for col in num_cols:
    try:
        skew = X[col].dropna().skew()
        if skew is not None and skew > 1:
            X[col] = np.log1p(X[col].clip(lower=0))
            skewed.append(col)
    except Exception:
        pass

print("NUM_COLS:", num_cols)
print("CAT_COLS:", cat_cols)
print("LOG1P_COLS:", skewed)
print("NULL_PCT_TOP5:")
null_rate = (df.isna().mean().sort_values(ascending=False).head(5) * 100).round(2)
for k, v in null_rate.items():
    print(f"  {{k}}: {{v}}%")
'''

    async def _build_training_code_with_llm(self, dataset_path: Path, rag_context: str) -> tuple[str, str, str]:
        prompt = (
            "You are an ML Engineer. Generate Python code ONLY (no explanations). "
            "Goal: train a model with feature engineering based on the provided context. "
            "Constraints:\n"
            "- Use pandas + scikit-learn only (no external downloads).\n"
            "- Read CSV at the provided dataset path.\n"
            "- Use model_transparency.infer_target_column to pick target.\n"
            "- Build a preprocessing pipeline (impute + one-hot for categoricals).\n"
            "- Apply log1p to skewed numeric features (skew > 1) if possible.\n"
            "- Train RandomForestClassifier or RandomForestRegressor depending on target cardinality.\n"
            "- Compute metrics. For regression, compute RMSE without using mean_squared_error(..., squared=False).\n"
            "- Save model to ml_engineer/model/model.joblib and metrics to shared/metrics.json and ml_engineer/metrics.json.\n"
            "- Print key steps and metrics.\n\n"
            f"DATASET_PATH: {dataset_path}\n\n"
            f"CONTEXT:\n{rag_context}\n"
        )
        out = await self._call_llm(prompt)
        if out.startswith("[LLM"):
            return "", out, out
        code = self._extract_code(out)
        if not code.strip():
            return "", "LLM: empty code", out
        return code, "LLM: success", out

    async def _fix_training_code_with_llm(
        self,
        dataset_path: Path,
        rag_context: str,
        current_code: str,
        error_output: str,
    ) -> tuple[str, str, str]:
        prompt = (
            "You are an ML Engineer. Fix the Python training code. Return ONLY the full corrected code.\n"
            "Constraints:\n"
            "- Keep pandas + scikit-learn only.\n"
            "- Use model_transparency.infer_target_column for target selection.\n"
            "- Ensure metrics are computed and printed.\n"
            "- For regression RMSE, do NOT use mean_squared_error(..., squared=False).\n\n"
            f"DATASET_PATH: {dataset_path}\n\n"
            f"CONTEXT:\n{rag_context}\n\n"
            f"ERROR OUTPUT:\n{error_output}\n\n"
            f"CURRENT CODE:\n{current_code}\n"
        )
        out = await self._call_llm(prompt)
        if out.startswith("[LLM"):
            return "", out, out
        code = self._extract_code(out)
        if not code.strip():
            return "", "LLM: empty code", out
        return code, "LLM: success", out

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
            ds_path = self.find_latest_uploaded_dataset()
            if not ds_path:
                await self.report(
                    "No uploaded dataset found in shared/datasets. Upload a file first.",
                    task_id,
                )
                return None

            rag_context = self._collect_rag_context()
            if workspace.is_initialized:
                workspace.write("ml_engineer", "rag_context.txt", rag_context or "[no rag context found]", task_id)

            llm_code, llm_status, llm_raw = await self._build_training_code_with_llm(ds_path, rag_context)
            used_llm = bool(llm_code.strip())
            code = llm_code or self._build_training_code(ds_path, rag_context)
            code, sanitize_notes = self._sanitize_training_code(code)
            retry_started = False
            syntax_ok, syntax_err = self._validate_python_code(code)
            if workspace.is_initialized:
                workspace.write("ml_engineer", "training_pipeline.py", code, task_id)
                fe_code = self._build_feature_engineering_code(ds_path)
                workspace.write("ml_engineer", "feature_engineering.py", fe_code, task_id)
                workspace.write(
                    "ml_engineer",
                    "llm_status.txt",
                    f"training_pipeline: {llm_status}\n"
                    f"feature_engineering: LLM disabled\n"
                    f"sanitize_notes: {sanitize_notes}\n"
                    f"initial_syntax_ok: {str(syntax_ok).lower()}\n"
                    f"initial_syntax_error: {syntax_err or '[none]'}\n",
                    task_id,
                )
                raw_payload = (
                    "## training_pipeline raw\n"
                    + (llm_raw or "[no raw response]")
                    + "\n\n## feature_engineering raw\n"
                    + "[LLM disabled]"
                    + "\n"
                )
                workspace.write("ml_engineer", "llm_raw.txt", raw_payload, task_id)

            await self.report(
                "ML Engineer training started in Jupyter kernel.\n"
                "Code saved: ml_engineer/training_pipeline.py\n"
                "Feature engineering saved: ml_engineer/feature_engineering.py\n"
                "RAG context saved: ml_engineer/rag_context.txt",
                task_id,
            )

            if used_llm and not syntax_ok:
                retry_started = True
                fix_code, fix_status, fix_raw = await self._fix_training_code_with_llm(
                    ds_path,
                    rag_context,
                    code,
                    syntax_err,
                )
                if fix_code.strip():
                    code = fix_code
                    if workspace.is_initialized:
                        workspace.write("ml_engineer", "training_pipeline.py", code, task_id)
                        workspace.append(
                            "ml_engineer",
                            "llm_status.txt",
                            f"training_pipeline_fix_0: {fix_status}\nretry_started: true",
                            task_id,
                        )
                        workspace.append(
                            "ml_engineer",
                            "llm_raw.txt",
                            "\n\n## training_pipeline fix raw\n" + (fix_raw or "[no raw response]"),
                            task_id,
                        )
                else:
                    if workspace.is_initialized:
                        workspace.append(
                            "ml_engineer",
                            "llm_status.txt",
                            "retry_started: true",
                            task_id,
                        )

            exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=600)
            if not exec_result.get("success") and used_llm:
                if workspace.is_initialized:
                    workspace.write(
                        "ml_engineer",
                        "last_exec_error.txt",
                        exec_result.get("output", "") or "[no output]",
                        task_id,
                    )
                    workspace.append(
                        "ml_engineer",
                        "llm_status.txt",
                        "retry_started: true",
                        task_id,
                    )
                fix_attempts_raw = (os.getenv("GROQ_CODE_FIX_RETRIES") or "3").strip()
                try:
                    fix_attempts = max(0, int(fix_attempts_raw))
                except Exception:
                    fix_attempts = 1
                for attempt in range(fix_attempts):
                    await self.report(
                        f"ML Engineer retry {attempt+1}/{fix_attempts}: sending error to LLM for fix.",
                        task_id,
                    )
                    retry_started = True
                    if workspace.is_initialized:
                        workspace.append(
                            "ml_engineer",
                            "llm_raw.txt",
                            f"\n\n## training_pipeline fix input error (attempt {attempt+1})\n"
                            + (exec_result.get("output", "") or "[no output]"),
                            task_id,
                        )
                    fix_code, fix_status, fix_raw = await self._fix_training_code_with_llm(
                        ds_path,
                        rag_context,
                        code,
                        exec_result.get("output", ""),
                    )
                    if workspace.is_initialized:
                        workspace.append(
                            "ml_engineer",
                            "llm_status.txt",
                            f"training_pipeline_fix_{attempt+1}: {fix_status}",
                            task_id,
                        )
                        workspace.append(
                            "ml_engineer",
                            "llm_raw.txt",
                            f"\n\n## training_pipeline fix raw (attempt {attempt+1})\n"
                            + (fix_raw or "[no raw response]"),
                            task_id,
                        )
                    if fix_code.strip():
                        code = fix_code
                        code, sanitize_fix_notes = self._sanitize_training_code(code)
                        used_llm = True
                        if workspace.is_initialized:
                            workspace.write("ml_engineer", "training_pipeline.py", code, task_id)
                            workspace.append(
                                "ml_engineer",
                                "llm_status.txt",
                                f"sanitize_notes_fix_{attempt+1}: {sanitize_fix_notes}",
                                task_id,
                            )
                        await self.report(
                            f"ML Engineer retry {attempt+1}/{fix_attempts}: running corrected code.",
                            task_id,
                        )
                    else:
                        if workspace.is_initialized:
                            workspace.append(
                                "ml_engineer",
                                "llm_status.txt",
                                f"training_pipeline_fix_{attempt+1}_note: empty code from LLM",
                                task_id,
                            )

                    exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=600)
                    if workspace.is_initialized:
                        workspace.write(
                            "ml_engineer",
                            "last_exec_error.txt",
                            exec_result.get("output", "") or "[no output]",
                            task_id,
                        )
                        workspace.append(
                            "ml_engineer",
                            "llm_raw.txt",
                            f"\n\n## training_pipeline fix error (attempt {attempt+1})\n"
                            + (exec_result.get("output", "") or "[no output]"),
                            task_id,
                        )
                    if exec_result.get("success"):
                        break
            elif workspace.is_initialized and used_llm:
                workspace.append(
                    "ml_engineer",
                    "llm_status.txt",
                    f"retry_started: {str(retry_started).lower()}",
                    task_id,
                )
            output = exec_result.get("output", "") or "[no output]"
            run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            if workspace.is_initialized:
                workspace.write("ml_engineer", "jupyter_output.txt", output, task_id)
                report = (
                    f"# ML Engineer Training Report\n\n"
                    f"Dataset: `{ds_path}`\n"
                    f"Runner: {exec_result.get('runner')}\n"
                    f"Timestamp: {run_ts}\n\n"
                    f"LLM used: {str(used_llm).lower()}\n\n"
                    f"Feature engineering script: `ml_engineer/feature_engineering.py`\n\n"
                    f"## RAG Context Used\n```\n{rag_context or '[none]'}\n```\n\n"
                    f"## Raw Execution Output\n```\n{output}\n```\n"
                )
                workspace.write("ml_engineer", "report.md", report, task_id)

            await self.report(
                "ML Engineer completed training.\n"
                "Saved: ml_engineer/training_pipeline.py, ml_engineer/feature_engineering.py, ml_engineer/jupyter_output.txt, ml_engineer/report.md\n"
                f"Execution runner: {exec_result.get('runner')} | success={exec_result.get('success')}",
                task_id,
            )
            await self.message(
                "orchestrator",
                "ML Engineer completed training. Artifacts and report.md are ready.",
                task_id,
            )
            return None

        await self.report("ML Engineer ready. Waiting for model-related instruction.", task_id)
        return None

    def get_tools(self):
        return []
