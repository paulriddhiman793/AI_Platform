from datetime import datetime
import json
import os
import time
import asyncio
import ast
import re
import math
import base64
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

    def _extract_json_block(self, output: str, begin: str, end: str) -> dict | None:
        """
        Extract and parse a JSON block delimited by begin/end markers from script output.
        Handles whitespace, encoding artifacts, NaN/Infinity literals, and trailing commas.
        """
        if not output or begin not in output or end not in output:
            return None
        blob = output.split(begin, 1)[1].split(end, 1)[0].strip()
        if not blob:
            return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            import re
            blob = re.sub(r",\s*([}\]])", r"\1", blob)     # trailing commas
            blob = re.sub(r"\bNaN\b",      "null",  blob)  # NaN      â†’ null
            blob = re.sub(r"\bInfinity\b", "1e308", blob)  # Infinity â†’ large float
            blob = re.sub(r"\b-Infinity\b","-1e308",blob)
            try:
                return json.loads(blob)
            except Exception:
                return None

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

    def _build_comparison_report(self, raw_payload: dict | None, eng_payload: dict | None,
                                 raw_path: Path | None, eng_path: Path | None) -> str:
        def _best_summary(p: dict | None) -> dict:
            if not p:
                return {}
            best = p.get("best_model") or {}
            task_type = p.get("task_type", "unknown")
            metric = "r2" if task_type == "regression" else "f1"
            test = best.get("test", {}) or {}
            score = best.get("score")
            if score is None:
                score = test.get(metric)
            return {
                "model": best.get("model"),
                "score": score,
                "gap": best.get("gap"),
                "underfit": best.get("underfit"),
                "task_type": task_type,
            }

        raw_s = _best_summary(raw_payload)
        eng_s = _best_summary(eng_payload)
        lines = [
            "=" * 72,
            "  ML ENGINEER - RAW VS ENGINEERED COMPARISON",
            "=" * 72,
            f"  Raw dataset       : {raw_path or 'N/A'}",
            f"  Engineered dataset: {eng_path or 'N/A'}",
            "",
        ]
        if raw_s:
            lines += [
                "RAW BEST MODEL",
                "-" * 72,
                f"  Model   : {raw_s.get('model')}",
                f"  Score   : {raw_s.get('score')}",
                f"  Gap     : {raw_s.get('gap')}",
                f"  Underfit: {raw_s.get('underfit')}",
                "",
            ]
        else:
            lines += ["RAW BEST MODEL", "-" * 72, "  No results.", ""]

        if eng_s:
            lines += [
                "ENGINEERED BEST MODEL",
                "-" * 72,
                f"  Model   : {eng_s.get('model')}",
                f"  Score   : {eng_s.get('score')}",
                f"  Gap     : {eng_s.get('gap')}",
                f"  Underfit: {eng_s.get('underfit')}",
                "",
            ]
        else:
            lines += ["ENGINEERED BEST MODEL", "-" * 72, "  No results.", ""]

        # Simple improvement summary
        try:
            if raw_s and eng_s:
                raw_score = float(raw_s.get("score"))
                eng_score = float(eng_s.get("score"))
                delta = eng_score - raw_score
                pct = (delta / abs(raw_score) * 100) if raw_score != 0 else None
                lines.append("SUMMARY")
                lines.append("-" * 72)
                if pct is not None:
                    lines.append(f"  Score change: {delta:.6f} ({pct:.2f}%)")
                else:
                    lines.append(f"  Score change: {delta:.6f}")
        except Exception:
            pass

        return "\n".join(lines)

    async def _run_training_for_dataset(self, ds_path: Path, label: str, task_id: str | None, target_col: str | None = None) -> dict | None:
        code = self._build_ml_optuna_code(Path(ds_path), label, target_col)
        exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=1200)
        output = exec_result.get("output", "") or "[no output]"
        if workspace.is_initialized:
            workspace.write("ml_engineer", f"jupyter_output_{label}.txt", output, task_id)

        payload = self._extract_json_block(output, "__ML_RESULTS_BEGIN__", "__ML_RESULTS_END__")
        if payload and workspace.is_initialized:
            workspace.write(
                "ml_engineer",
                f"ml_results_{label}.json",
                json.dumps(payload, indent=2, default=str),
                task_id,
            )
        if payload:
            report, final_report = self._build_ml_reports(Path(ds_path), payload)
        else:
            report = output
            final_report = "Best model could not be determined (no structured results)."

        if workspace.is_initialized:
            workspace.write("ml_engineer", f"report_{label}.txt", report, task_id)
            workspace.write("ml_engineer", f"final_report_{label}.txt", final_report, task_id)
        return payload

    def _load_ml_results(self, label: str) -> dict | None:
        if not workspace.is_initialized or not workspace.project_root:
            return None
        root = workspace.project_root
        p = root / "ml_engineer" / f"ml_results_{label}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        # Fallback: parse jupyter output
        jp = root / "ml_engineer" / f"jupyter_output_{label}.txt"
        if jp.exists():
            try:
                output = jp.read_text(encoding="utf-8", errors="replace")
                return self._extract_json_block(output, "__ML_RESULTS_BEGIN__", "__ML_RESULTS_END__")
            except Exception:
                return None
        return None

    def _local_worker_api_base(self) -> str:
        base = (os.getenv("API_URL") or "").strip()
        if not base:
            port = (os.getenv("PORT") or "8000").strip()
            base = f"http://127.0.0.1:{port}"
        return base.rstrip("/")

    async def _local_worker_exec(self, auth_token: str, command: str, detach: bool = True, cwd: str | None = None) -> dict:
        import requests

        url = f"{self._local_worker_api_base()}/worker/exec"
        timeout_s = 150

        def _call():
            return requests.post(
                url,
                json={
                    "auth_token": auth_token,
                    "command": command,
                    "detach": detach,
                    "cwd": cwd or "",
                },
                timeout=timeout_s,
            )

        resp = await asyncio.to_thread(_call)
        data = resp.json() if resp.content else {}
        if not resp.ok:
            raise RuntimeError(data.get("detail") or f"Worker exec failed ({resp.status_code})")
        return data

    async def _local_worker_write_file(self, auth_token: str, rel_path: str, content: str, cwd: str | None = None) -> dict:
        import requests

        url = f"{self._local_worker_api_base()}/worker/write_file"
        timeout_s = 150
        payload = {
            "auth_token": auth_token,
            "path": rel_path,
            "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "cwd": cwd or "",
        }

        def _call():
            return requests.post(url, json=payload, timeout=timeout_s)

        resp = await asyncio.to_thread(_call)
        data = resp.json() if resp.content else {}
        if not resp.ok:
            raise RuntimeError(data.get("detail") or f"Worker write_file failed ({resp.status_code})")
        return data

    def _serve_model_script(self) -> str:
        try:
            p = Path(__file__).resolve().parent.parent / "tools" / "serve_model.py"
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    async def _write_file_via_exec_chunks(self, auth_token: str, rel_path: str, content: str, cwd: str | None) -> None:
        data = content.encode("utf-8")
        init_cmd = (
            "python -c \"from pathlib import Path; "
            f"p=Path(r'{rel_path}'); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_bytes(b'')\""
        )
        await self._local_worker_exec(auth_token, init_cmd, detach=False, cwd=cwd)
        chunk_size = 1200
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode("ascii")
            append_cmd = (
                "python -c \"import base64; from pathlib import Path; "
                f"p=Path(r'{rel_path}'); "
                f"p.open('ab').write(base64.b64decode('{chunk_b64}'))\""
            )
            result = await self._local_worker_exec(auth_token, append_cmd, detach=False, cwd=cwd)
            exec_result = result.get("result") or {}
            if int(exec_result.get("returncode", 1)) != 0:
                raise RuntimeError(exec_result.get("stderr") or "Chunked file write failed")

    async def _ensure_local_serve_script(self, auth_token: str, cwd: str | None) -> None:
        script = self._serve_model_script()
        if not script:
            return
        try:
            result = await self._local_worker_write_file(
                auth_token=auth_token,
                rel_path="ml_engineer/serve_model.py",
                content=script,
                cwd=cwd,
            )
            write_result = result.get("result") or {}
            if int(write_result.get("returncode", 1)) != 0:
                raise RuntimeError(write_result.get("stderr") or "Failed to write serve_model.py")
        except Exception:
            await self._write_file_via_exec_chunks(
                auth_token=auth_token,
                rel_path="ml_engineer/serve_model.py",
                content=script,
                cwd=cwd,
            )

    def _safe_model_name(self, name: str) -> str:
        if not name:
            return ""
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_").lower()

    def _model_aliases(self, name: str) -> set[str]:
        raw = self._safe_model_name(name)
        if not raw:
            return set()
        norm = re.sub(r"[^a-z0-9]+", "", raw)
        aliases = {raw, norm}
        trimmed = norm.replace("regressor", "").replace("classifier", "")
        if trimmed:
            aliases.add(trimmed)
        alias_map = {
            "xgboost": {"xgb", "xgboost", "xgbregressor", "xgbclassifier"},
            "xgb": {"xgb", "xgboost", "xgbregressor", "xgbclassifier"},
            "lightgbm": {"lgbm", "lightgbm", "lgbmregressor", "lgbmclassifier"},
            "lgbm": {"lgbm", "lightgbm", "lgbmregressor", "lgbmclassifier"},
            "randomforest": {"randomforest", "randomforestregressor", "randomforestclassifier"},
            "gradientboosting": {"gradientboosting", "gradientboostingregressor", "gradientboostingclassifier"},
            "linear": {"linear", "linearregression", "logisticregression"},
            "catboost": {"catboost", "catboostregressor", "catboostclassifier"},
            "ridge": {"ridge"},
            "lasso": {"lasso"},
            "svm": {"svm", "svr", "svc"},
            "svr": {"svm", "svr"},
            "svc": {"svm", "svc"},
            "knn": {"knn", "kneighbors", "kneighborsregressor", "kneighborsclassifier"},
        }
        for key, vals in alias_map.items():
            if key in aliases or key in norm:
                aliases.update(vals)
        return {a for a in aliases if a}

    def _extract_model_hint(self, msg: str) -> str | None:
        known = [
            "xgboost", "xgb", "lightgbm", "lgbm", "catboost", "randomforest",
            "gradientboosting", "ridge", "lasso", "linear", "elasticnet",
            "svm", "svr", "knn",
        ]
        for k in known:
            if k in msg:
                return k
        m = re.search(r"model\s*[:\-]?\s*(\w+)", msg, re.I)
        if m:
            return m.group(1)
        return None

    def _resolve_local_model_path(self, base_dir: Path, model_hint: str | None, dataset_tag: str | None) -> Path | None:
        tags = []
        if dataset_tag:
            tags.append(dataset_tag)
        tags += ["engineered", "raw"]
        seen = set()
        hint_aliases = self._model_aliases(model_hint or "")
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            models_dir = base_dir / "ml_engineer" / f"models_{tag}"
            if not models_dir.exists():
                continue
            files = list(models_dir.glob("*.joblib"))
            if not files:
                continue
            if hint_aliases:
                for f in files:
                    stem_aliases = self._model_aliases(f.stem)
                    if hint_aliases.intersection(stem_aliases):
                        return f
            # fallback: latest
            return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        # fallback to single model path
        fallback = base_dir / "ml_engineer" / "model" / "model.joblib"
        if fallback.exists():
            return fallback
        fallback = base_dir / "shared" / "model.joblib"
        if fallback.exists():
            return fallback
        return None

    def _find_dataset_from_message(self, msg: str) -> tuple[Path | None, str | None]:
        """
        Return (path, label) where label is 'raw' or 'engineered' when possible.
        """
        if not workspace.is_initialized or not workspace.project_root:
            return None, None
        root = workspace.project_root
        ds_dir = root / "shared" / "datasets"
        if not ds_dir.exists():
            return None, None

        m = re.search(r"([A-Za-z0-9_.-]+\\.(?:csv|xlsx|xls|parquet|json|tsv))", msg, re.I)
        if m:
            name = m.group(1)
            candidate = ds_dir / name
            if candidate.exists():
                label = "engineered" if "engineered" in candidate.name.lower() else "raw"
                return candidate, label

        if "engineer" in msg or "engineered" in msg:
            p = self.find_latest_engineered_dataset()
            return (Path(p) if p else None), "engineered"
        if "raw" in msg or "original" in msg or "baseline" in msg:
            p = self._find_latest_raw_dataset()
            return (Path(p) if p else None), "raw"

        # Default preference: engineered if available
        p = self.find_latest_engineered_dataset()
        if p:
            return Path(p), "engineered"
        p = self._find_latest_raw_dataset()
        return (Path(p) if p else None), ("raw" if p else None)

    def _match_model_in_results(self, query: str, payload: dict | None) -> dict | None:
        if not payload:
            return None
        results = payload.get("results", [])
        if not results:
            return None
        q = query.lower()
        q_norm = re.sub(r"[^a-z0-9]+", "", q)

        def _norm(name: str) -> str:
            n = name.lower()
            n = n.replace("regressor", "").replace("classifier", "")
            n = re.sub(r"[^a-z0-9]+", "", n)
            return n

        # Direct match against model names
        for r in results:
            name = str(r.get("model", "")).lower()
            if q == name:
                return r
        # Substring match
        for r in results:
            name = str(r.get("model", "")).lower()
            if q in name:
                return r
        # Normalized match (allows omitting regressor/classifier)
        for r in results:
            name = str(r.get("model", ""))
            if _norm(name) and _norm(name) in q_norm:
                return r
        # Keyword mapping
        aliases = {
            "xgb": "xgb",
            "xgboost": "xgb",
            "lgb": "lgbm",
            "lgbm": "lgbm",
            "lightgbm": "lgbm",
            "catboost": "catboost",
            "random forest": "randomforest",
            "randomforest": "randomforest",
            "gradient boosting": "gradientboost",
            "linear": "linearregression",
            "ridge": "ridge",
            "lasso": "lasso",
            "elastic": "elasticnet",
            "svm": "svr",
            "svr": "svr",
            "svc": "svc",
            "knn": "kneighbors",
            "k-nearest": "kneighbors",
            "decision tree": "decisiontree",
        }
        for k, v in aliases.items():
            if k in q:
                for r in results:
                    name = str(r.get("model", "")).lower()
                    if v in name:
                        return r
        return None

    def _model_import_block(self, model_name: str) -> str:
        model = model_name
        if model in ("XGBRegressor", "XGBClassifier"):
            return "from xgboost import XGBRegressor, XGBClassifier"
        if model in ("LGBMRegressor", "LGBMClassifier"):
            return "from lightgbm import LGBMRegressor, LGBMClassifier"
        if model in ("CatBoostRegressor", "CatBoostClassifier"):
            return "from catboost import CatBoostRegressor, CatBoostClassifier"
        mapping = {
            "LinearRegression": "from sklearn.linear_model import LinearRegression",
            "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
            "Ridge": "from sklearn.linear_model import Ridge",
            "Lasso": "from sklearn.linear_model import Lasso",
            "ElasticNet": "from sklearn.linear_model import ElasticNet",
            "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
            "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
            "GradientBoostingRegressor": "from sklearn.ensemble import GradientBoostingRegressor",
            "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
            "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor",
            "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier",
            "SVR": "from sklearn.svm import SVR",
            "SVC": "from sklearn.svm import SVC",
            "KNeighborsRegressor": "from sklearn.neighbors import KNeighborsRegressor",
            "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
            "HistGradientBoostingRegressor": "from sklearn.ensemble import HistGradientBoostingRegressor",
            "HistGradientBoostingClassifier": "from sklearn.ensemble import HistGradientBoostingClassifier",
        }
        return mapping.get(model, "")

    def _should_scale_for_model(self, model_name: str) -> bool:
        name = model_name.lower()
        if any(k in name for k in ("linear", "ridge", "lasso", "elastic", "logistic", "svr", "svc", "knn")):
            return True
        return False

    def _build_transparency_code(self, dataset_path: Path, model_name: str,
                                 params: dict, task_type: str, target_col: str | None) -> str:
        ds = str(dataset_path).replace("\\", "\\\\")
        import_stmt = self._model_import_block(model_name)
        scale_flag = "True" if self._should_scale_for_model(model_name) else "False"
        target_line = f"target_col = {json.dumps(target_col)}" if target_col else "target_col = None"
        params_literal = json.dumps(params or {}, indent=2)
        return f'''import json
import pandas as pd
from model_transparency import run_pipeline_from_df, infer_target_column
{import_stmt}

df = pd.read_csv(r"{ds}")
{target_line}
if target_col is None:
    target_col, _, _, _ = infer_target_column(df, verbose=False)

params = {params_literal}
model_name = "{model_name}"

try:
    model = {model_name}(**params)
except Exception as e:
    raise SystemExit(f"Failed to instantiate {{model_name}} with params {{params}}: {{e}}")

print("TRAINING_PROCESS_BEGIN")
print(f"MODEL: {{model_name}}")
print(f"DATASET: {ds}")
print(f"TARGET: {{target_col}}")
print(f"TASK: {task_type}")
print(f"PARAMS: {{json.dumps(params, indent=2)}}")
print("TRAINING_PROCESS_END")

run_pipeline_from_df(
    model, df,
    target_col=target_col,
    task_type="{task_type}",
    test_size=0.2,
    scale={scale_flag},
    cv=5,
    n_walkthrough=5,
    disable_catboost_swap=True,
)
'''

    def _next_training_process_name(self) -> str:
        if not workspace.is_initialized or not workspace.project_root:
            return "training_process.txt"
        root = workspace.project_root / "ml_engineer"
        base = root / "training_process.txt"
        if not base.exists():
            return "training_process.txt"
        idx = 1
        while True:
            candidate = root / f"training_process_{idx}.txt"
            if not candidate.exists():
                return candidate.name
            idx += 1

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_ml_reports(self, dataset_path: "Path", payload: dict) -> tuple[str, str]:
        """
        Build two plain-text reports from the v2 ML pipeline payload.

        Report 1 â€” Full model comparison:
            â€¢ Every model's train/test/CV metrics, overfit gap, stability, timing
            â€¢ Leakage / suspect feature audit section
            â€¢ Ranked leaderboard table with family and CV-std columns

        Report 2 â€” Final best-model report:
            â€¢ Performance summary with score breakdown (cv_mean, std penalty, overfit penalty)
            â€¢ All tuned hyperparameters
            â€¢ SHAP top-15 feature importances (ASCII bar chart)
            â€¢ Leakage warnings surfaced prominently if any features were removed
        """
        task_type   = payload.get("task_type",   "unknown")
        target_col  = payload.get("target_col",  "unknown")
        n_train     = payload.get("n_train",     "?")
        n_test      = payload.get("n_test",      "?")
        n_folds     = payload.get("n_cv_folds",  "?")
        optuna_t    = payload.get("optuna_trials","?")
        timestamp   = payload.get("run_timestamp","unknown")
        results     = payload.get("results",     [])
        leaderboard = payload.get("leaderboard", [])
        best        = payload.get("best_model")
        shap_fi     = payload.get("feature_importances_shap", {})
        leaky       = payload.get("leaky_features_removed",   [])
        suspect     = payload.get("suspect_features",         [])
        feat_corr   = payload.get("feature_target_corr",      {})

        primary = "r2" if task_type == "regression" else "f1"

        # â”€â”€ Report 1: full comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lines = [
            "=" * 72,
            "  ML ENGINEER â€” FULL MODEL COMPARISON REPORT  (v2 leakage-aware)",
            "=" * 72,
            f"  Run timestamp : {timestamp}",
            f"  Dataset       : {dataset_path}",
            f"  Target column : {target_col}",
            f"  Task type     : {task_type}",
            f"  Train / Test  : {n_train} / {n_test} samples",
            f"  CV folds      : {n_folds}",
            f"  Optuna trials : {optuna_t}",
            "=" * 72,
            "",
        ]

        # â”€â”€ Feature audit section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if feat_corr or leaky or suspect:
            lines += ["FEATURE AUDIT", "-" * 72]
            if leaky:
                lines.append(f"  âš  LEAKAGE â€” features REMOVED before training (|r| â‰¥ 0.98):")
                for f in leaky:
                    r_val = feat_corr.get(f)
                    r_str = f"{r_val:+.6f}" if r_val is not None else "N/A"
                    lines.append(f"      {f:<40}  r = {r_str}")
                lines.append("")
            if suspect:
                lines.append(f"  âš¡ SUSPECT â€” high correlation kept (0.95 â‰¤ |r| < 0.98):")
                for f in suspect:
                    r_val = feat_corr.get(f)
                    r_str = f"{r_val:+.6f}" if r_val is not None else "N/A"
                    lines.append(f"      {f:<40}  r = {r_str}")
                lines.append("")
            if feat_corr:
                lines.append("  Featureâ€“target correlations (top 20 by |r|):")
                sorted_corr = sorted(
                    [(k, v) for k, v in feat_corr.items() if v is not None],
                    key=lambda x: abs(x[1]), reverse=True,
                )
                for fname, rval in sorted_corr[:20]:
                    r_abs = abs(rval) if (rval is not None and math.isfinite(rval)) else 0.0
                    bar = "â–ˆ" * max(1, int(r_abs * 20))
                    tag = "  âš  LEAKY"   if fname in leaky   else \
                          "  âš¡ suspect" if fname in suspect else ""
                    lines.append(f"    {fname:<38}  {rval:+.4f}  {bar}{tag}")
                if len(sorted_corr) > 20:
                    lines.append(f"    â€¦ and {len(sorted_corr)-20} more features")
            lines.append("")

        # â”€â”€ Per-model results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lines += ["MODEL RESULTS", "-" * 72]

        for r in results:
            name   = r.get("model",  "unknown")
            status = r.get("status", "error")
            family = r.get("family", "")

            if status != "ok":
                lines.append(f"  âœ— {name}")
                lines.append(f"      ERROR: {r.get('error', 'unknown error')}")
                lines.append("")
                continue

            train_m  = r.get("train", {})
            test_m   = r.get("test",  {})
            cv_mean  = r.get("cv_test_mean")
            cv_std   = r.get("cv_test_std")
            gap      = r.get("gap")
            underfit = r.get("underfit", False)
            score    = r.get("score")
            elapsed  = r.get("elapsed_sec")
            params   = r.get("best_params", {})

            cv_str  = f"{cv_mean:.4f} Â± {cv_std:.4f}" if cv_mean is not None else "N/A"
            gap_str = f"{gap:.4f}" if gap is not None else "N/A"
            sc_str  = f"{score:.4f}" if score is not None else "N/A"
            t_str   = f"{elapsed:.1f}s" if elapsed is not None else "N/A"

            overfit_warn = "  âš  overfit"  if gap    is not None and gap    > 0.10 else ""
            stab_warn    = "  âš  unstable" if cv_std is not None and cv_std > 0.01 else ""

            lines.append(f"  â–º {name}  [{family}]")
            lines.append(f"      Train metrics  : {train_m}")
            lines.append(f"      Test  metrics  : {test_m}")
            lines.append(f"      CV {primary:>8}    : {cv_str}{stab_warn}")
            lines.append(f"      Overfit gap    : {gap_str}{overfit_warn}")
            lines.append(f"      Underfit       : {'âš  YES' if underfit else 'No'}")
            lines.append(f"      Composite score: {sc_str}")
            lines.append(f"      Training time  : {t_str}")
            if params:
                lines.append(f"      Best params    : {params}")
            lines.append("")

        # â”€â”€ Leaderboard table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if leaderboard:
            lines += [
                "-" * 72,
                "  LEADERBOARD  (score = cv_mean âˆ’ 0.5Ã—cv_std âˆ’ overfit_penalty)",
                "-" * 72,
                f"  {'Rank':<5} {'Model':<28} {'Family':<10} "
                f"{'CV mean':>9} {'CV std':>8} {'Hold-out':>9} {'Score':>8}",
                f"  {'-'*5} {'-'*28} {'-'*10} {'-'*9} {'-'*8} {'-'*9} {'-'*8}",
            ]
            for rank, r in enumerate(leaderboard, start=1):
                if r.get("status") != "ok":
                    continue
                cv_m = r.get("cv_test_mean")
                cv_s = r.get("cv_test_std")
                ho   = r.get("test", {}).get(primary)
                sc   = r.get("score")
                fam  = r.get("family", "")
                if all(v is not None for v in [cv_m, cv_s, ho, sc]):
                    lines.append(
                        f"  {rank:<5} {r['model']:<28} {fam:<10} "
                        f"{cv_m:>9.4f} {cv_s:>8.4f} {ho:>9.4f} {sc:>8.4f}"
                    )
                else:
                    lines.append(f"  {rank:<5} {r['model']:<28}  (metrics unavailable)")
            lines.append("")

        report = "\n".join(lines)

        # â”€â”€ Report 2: final best-model report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        final_lines = [
            "=" * 72,
            "  ML ENGINEER â€” FINAL MODEL REPORT  (v2 leakage-aware)",
            "=" * 72,
            f"  Run timestamp : {timestamp}",
            f"  Dataset       : {dataset_path}",
            f"  Target column : {target_col}",
            f"  Task type     : {task_type}",
            f"  Train / Test  : {n_train} / {n_test} samples",
            f"  CV strategy   : {n_folds}-fold cross-validation",
            "=" * 72,
            "",
        ]

        # Leakage summary at top of final report
        if leaky:
            final_lines += [
                "  âš   DATA LEAKAGE DETECTED & RESOLVED",
                "     The following features were near-perfect linear predictors",
                "     of the target and were REMOVED before training.  Results",
                "     below reflect the clean, leakage-free dataset.",
                "",
            ]
            for f in leaky:
                r_val = feat_corr.get(f)
                final_lines.append(
                    f"     â€¢ {f}  (|r| = {abs(r_val):.6f})"
                    if r_val is not None else f"     â€¢ {f}"
                )
            final_lines.append("")

        if best:
            train_m  = best.get("train", {})
            test_m   = best.get("test",  {})
            cv_mean  = best.get("cv_test_mean")
            cv_std   = best.get("cv_test_std")
            gap      = best.get("gap")
            underfit = best.get("underfit", False)
            score    = best.get("score")
            params   = best.get("best_params", {})
            family   = best.get("family", "")

            cv_str   = f"{cv_mean:.4f} Â± {cv_std:.4f}" if cv_mean is not None else "N/A"
            gap_flag = "âš  overfitting detected"  if gap    and gap    > 0.10 else "âœ“ within range"
            uf_flag  = "âš  underfitting detected" if underfit                  else "âœ“ sufficient"
            stb_flag = "âš  high variance"         if cv_std and cv_std > 0.01 else "âœ“ stable"

            std_pen     = round(0.5 * (cv_std  or 0.0), 6)
            overfit_pen = round(max(0.0, (gap or 0.0) - 0.10) + (0.10 if underfit else 0.0), 6)
            cv_contrib  = round(cv_mean or 0.0, 6)

            final_lines += [
                f"  BEST MODEL : {best.get('model')}  [{family}]",
                "",
                "  Performance Summary",
                "  " + "-" * 44,
                f"  Train metrics       : {train_m}",
                f"  Test  metrics       : {test_m}",
                f"  CV {primary} (mean Â± std) : {cv_str}  â†’  {stb_flag}",
                (f"  Overfit gap         : {gap:.4f}  {gap_flag}"
                 if gap is not None else "  Overfit gap: N/A"),
                f"  Fit quality         : {uf_flag}",
                "",
                "  Score Breakdown",
                "  " + "-" * 44,
                f"    CV mean                       {cv_contrib:>+10.6f}",
                f"    âˆ’ stability penalty (0.5Ã—Ïƒ)  {-std_pen:>+10.6f}",
                f"    âˆ’ overfit/underfit penalty   {-overfit_pen:>+10.6f}",
                f"    {'â”€'*38}",
                (f"    Composite score               {score:>+10.6f}"
                 if score is not None else "    Composite score: N/A"),
                "",
                "  Hyperparameters",
                "  " + "-" * 44,
            ]
            if params:
                for k, v in params.items():
                    final_lines.append(f"    {k:<36} = {v}")
            else:
                final_lines.append("    (default / no tuning applied)")

            if shap_fi:
                max_imp = max(shap_fi.values()) if shap_fi else 1.0
                if not math.isfinite(max_imp):
                    max_imp = 1.0
                final_lines += [
                    "",
                    "  SHAP Feature Importances  (top 15, mean |SHAP| on test set)",
                    "  " + "-" * 44,
                ]
                for i, (feat, imp) in enumerate(list(shap_fi.items())[:15], start=1):
                    imp_val = imp if math.isfinite(imp) else 0.0
                    bar = "â–ˆ" * max(1, int(imp_val / max_imp * 24))
                    tag = "  âš¡ suspect" if feat.split("__")[-1] in suspect else ""
                    final_lines.append(f"    {i:>2}. {feat:<36} {imp:>8.4f}  {bar}{tag}")

        else:
            final_lines += [
                "  âœ— Best model could not be determined â€” all models failed.",
                "    Check ml_pipeline.log for details.",
            ]

        final_lines += ["", "=" * 72]
        final_report = "\n".join(final_lines)
        return report, final_report

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _build_ml_optuna_code(self, dataset_path: "Path", label: str, target_col: str | None = None) -> str:
        """
        Generate the full v2 ML pipeline script as a string, with the dataset
        path injected.  The generated script includes:

          â€¢ Automatic data leakage detection & soft-removal (|r| >= 0.98)
          â€¢ IQR x 3 outlier capping on numeric columns
          â€¢ Dual preprocessors: StandardScaler for linear, raw for trees
          â€¢ OHE for low-cardinality cats, OrdinalEncoder for high-card cats
          â€¢ 5-fold CV with no leakage (clone(pre) inside every fold)
          â€¢ Optuna HPO with MedianPruner (50 trials per model)
          â€¢ min_child_samples / min_child_weight in boosting search spaces
          â€¢ Dynamic MAX_LEAVES cap based on training set size (LGBM overfit fix)
          â€¢ Per-model family tags for diverse stacking
          â€¢ DIVERSE stacking ensemble (best-per-family, not top-3 same family)
          â€¢ Composite score = cv_mean - 0.5 x cv_std - overfit_penalty
          â€¢ Post-training sanity checks (perfect train R2, high relative RMSE)
          â€¢ SHAP feature importances for best model
          â€¢ Full structured logging to ml_pipeline.log
          â€¢ All results + audit data saved to ml_results.json
        """
        ds = str(dataset_path).replace("\\", "\\\\")
        target_line = f"target_col = {json.dumps(target_col)}" if target_col else "target_col = _infer_target(df)"
        return f'''"""
Advanced ML Pipeline v2 â€” Leakage-Aware, Stability-Penalised, Diversity-Enforced
==================================================================================
Auto-generated by _build_ml_optuna_code
Dataset : {ds}
"""

import os
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
import re
import joblib

from pathlib import Path
from datetime import datetime
import math

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, make_scorer
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.base import clone

warnings.filterwarnings("ignore")

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LOG_PATH = Path("ml_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RANDOM_SEED          = 42
np.random.seed(RANDOM_SEED)
N_CV_FOLDS           = 5
OPTUNA_TRIALS        = int(os.getenv("ML_OPTUNA_TRIALS", "50"))
HIGH_CARD_THR        = 15
IQR_CAP_FACTOR       = 3.0
ENABLE_STACKING      = True
ENABLE_SHAP          = True
LEAKAGE_CORR_THRESH  = 0.98
PERFECT_R2_THRESH    = 0.9999
CV_STD_PENALTY_COEFF = 0.5
N_JOBS               = int(os.getenv("ML_N_JOBS", "1"))

# â”€â”€ Target inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from model_transparency import infer_target_column
    _HAS_TRANSPARENCY = True
except ImportError:
    _HAS_TRANSPARENCY = False

def _infer_target(df):
    if _HAS_TRANSPARENCY:
        col, conf, reason, _ = infer_target_column(df, verbose=False)
        if col:
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = None
            conf_str = f"{{conf_val:.2f}}" if conf_val is not None else str(conf)
            log.info(f"Inferred target: '{{col}}' (confidence={{conf_str}}) — {{reason}}")
            return col
    col = df.columns[-1]
    log.warning(f"model_transparency not available; defaulting to last column '{{col}}'.")
    return col

# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH = r"{ds}"
MODEL_TAG = "{label}"
log.info(f"Loading dataset: {{DATA_PATH}}")
df = pd.read_csv(DATA_PATH)
log.info(f"Shape: {{df.shape}}  |  Columns: {{list(df.columns)}}")

{target_line}
if target_col not in df.columns:
    log.warning(f"Configured target '{{target_col}}' not found in dataset; falling back to auto-detection.")
    target_col = _infer_target(df)
y = df[target_col].copy()
X = df.drop(columns=[target_col]).copy()

n_unique  = y.nunique(dropna=True)
task_type = "classification" if n_unique <= 20 else "regression"
log.info(f"Task: {{task_type}}  |  Target unique values: {{n_unique}}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  FEATURE AUDIT â€” leakage detection via correlation with target
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
num_cols_raw = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_raw = [c for c in X.columns if c not in num_cols_raw]

for col in cat_cols_raw:
    try:
        X[col] = X[col].astype("string")
    except Exception:
        X[col] = X[col].astype(str)

leaky_features   = []
suspect_features = []
feature_corr     = {{}}

if task_type == "regression":
    log.info("â”€â”€ Feature Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    for col in num_cols_raw:
        try:
            r = float(np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1])
            feature_corr[col] = round(r, 6)
            abs_r = abs(r)
            if abs_r >= LEAKAGE_CORR_THRESH:
                leaky_features.append(col)
                log.warning(f"  âš  LEAKAGE  {{col:<35}} |r| = {{abs_r:.6f}}")
            elif abs_r >= 0.95:
                suspect_features.append(col)
                log.info(f"  âš¡ Suspect  {{col:<35}} |r| = {{abs_r:.6f}}")
        except Exception:
            feature_corr[col] = None

    if leaky_features:
        log.warning(
            f"\\n{'!'*60}\\n"
            f"  DATA LEAKAGE: {{len(leaky_features)}} feature(s) REMOVED\\n"
            f"    {{leaky_features}}\\n"
            f"{'!'*60}"
        )
        X = X.drop(columns=leaky_features)
        num_cols_raw = [c for c in num_cols_raw if c not in leaky_features]

# â”€â”€ Outlier capping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def cap_outliers(df_in, cols, factor=IQR_CAP_FACTOR):
    df_out = df_in.copy()
    for c in cols:
        q1, q3 = df_out[c].quantile(0.25), df_out[c].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            df_out[c] = df_out[c].clip(q1 - factor * iqr, q3 + factor * iqr)
    return df_out

X = cap_outliers(X, num_cols_raw)
log.info(f"Outlier capping: {{len(num_cols_raw)}} numeric cols (IQRÃ—{{IQR_CAP_FACTOR}})")

# â”€â”€ Categorical cardinality split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
low_cat  = [c for c in cat_cols_raw if X[c].nunique() <= HIGH_CARD_THR]
high_cat = [c for c in cat_cols_raw if X[c].nunique() >  HIGH_CARD_THR]
log.info(f"Numeric: {{len(num_cols_raw)}} | Low cats: {{len(low_cat)}} | High cats: {{len(high_cat)}}")

# â”€â”€ Preprocessors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [
        ("num",      Pipeline(num_steps),                                        num_cols_raw),
        ("low_cat",  Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot",  _make_ohe())]),                       low_cat),
        ("high_cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("ordinal", OrdinalEncoder(
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1))]),                          high_cat),
    ]
    transformers = [(n, t, c) for n, t, c in transformers if len(c) > 0]
    return ColumnTransformer(transformers=transformers, remainder="drop")

pre_scaled   = build_preprocessor(scale_numeric=True)
pre_unscaled = build_preprocessor(scale_numeric=False)

# â”€â”€ Train / test split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED,
    stratify=y if task_type == "classification" else None,
)
log.info(f"Train: {{len(X_train)}}  |  Test: {{len(X_test)}}")

MAX_LEAVES  = min(256, max(16, len(X_train) // 20))
cv_splitter = (
    StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    if task_type == "classification"
    else KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
)
PRIMARY_METRIC = "r2" if task_type == "regression" else "f1_weighted"

# â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def eval_metrics(y_true, preds):
    if task_type == "regression":
        mse = mean_squared_error(y_true, preds)
        return {{"rmse": float(mse**0.5),
                 "mae":  float(mean_absolute_error(y_true, preds)),
                 "r2":   float(r2_score(y_true, preds))}}
    return {{"accuracy": float(accuracy_score(y_true, preds)),
             "f1":       float(f1_score(y_true, preds, average="weighted"))}}

def primary(m):
    return m.get("r2", m.get("f1", -1.0))

def composite_score(cv_mean, cv_std, train_m, test_m):
    tm, vm   = primary(train_m), primary(test_m)
    underfit = (task_type == "regression"    and tm < 0.6 and vm < 0.6) or \\
               (task_type == "classification" and tm < 0.7 and vm < 0.7)
    gap      = float(tm - vm) if (tm is not None and vm is not None) else 0.0
    penalty  = max(0.0, gap - 0.10) + (0.10 if underfit else 0.0)
    std_pen  = CV_STD_PENALTY_COEFF * (cv_std or 0.0)
    return (cv_mean or 0.0) - std_pen - penalty, gap, underfit, penalty

def sanity_check(name, train_m, test_m, y_arr):
    tp = primary(train_m)
    if tp is not None and tp > PERFECT_R2_THRESH:
        log.warning(f"  âš  PERFECT TRAIN FIT [{{name}}]: {{PRIMARY_METRIC}}={{tp:.8f}} â€” "
                    f"possible residual leakage.")
    if task_type == "regression":
        rmse, y_mean = test_m.get("rmse", 0), float(np.mean(np.abs(y_arr)))
        if y_mean > 0 and rmse / y_mean > 0.5:
            log.warning(f"  âš  HIGH RELATIVE ERROR [{{name}}]: RMSE={{rmse:.2f}} "
                        f"mean|y|={{y_mean:.2f}} ({{rmse/y_mean:.1%}})")

# â”€â”€ Optuna â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import optuna
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.warning("Optuna not installed â€” default hyperparameters only")

results = []

# â”€â”€ Model runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_model(name, builder, space_fn=None, use_scaling=False, family="other"):
    log.info(f"â–¶ {{name}}  [{{family}}]")
    t0  = time.time()
    pre = pre_scaled if use_scaling else pre_unscaled
    try:
        best_params = {{}}
        if HAS_OPTUNA and space_fn is not None:
            X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
                X_train, y_train, test_size=0.20, random_state=RANDOM_SEED,
                stratify=y_train if task_type == "classification" else None,
            )
            def objective(trial):
                params = space_fn(trial)
                pipe   = Pipeline([("pre", clone(pre)), ("model", builder(params))])
                pipe.fit(X_tr2, y_tr2)
                return primary(eval_metrics(y_val2, pipe.predict(X_val2)))
            study = optuna.create_study(
                direction="maximize",
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )
            study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
            best_params = study.best_params
            log.info(f"   Best params: {{best_params}}")

        scorer  = (make_scorer(r2_score) if task_type == "regression"
                   else make_scorer(f1_score, average="weighted"))
        cv_pipe = Pipeline([("pre", clone(pre)), ("model", builder(best_params))])
        cv_res  = cross_validate(cv_pipe, X_train, y_train, cv=cv_splitter,
                                 scoring=scorer, return_train_score=True, n_jobs=N_JOBS)
        cv_mean = float(np.mean(cv_res["test_score"]))
        cv_std  = float(np.std(cv_res["test_score"]))
        cv_tr   = float(np.mean(cv_res["train_score"]))
        log.info(f"   CV: {{cv_mean:.4f}} Â± {{cv_std:.4f}}  (train={{cv_tr:.4f}})")

        final_pipe = Pipeline([("pre", clone(pre)), ("model", builder(best_params))])
        final_pipe.fit(X_train, y_train)
        test_m  = eval_metrics(y_test,  final_pipe.predict(X_test))
        train_m = eval_metrics(y_train, final_pipe.predict(X_train))
        sanity_check(name, train_m, test_m, y_test.values)

        score, gap, underfit, penalty = composite_score(cv_mean, cv_std, train_m, test_m)
        elapsed = time.time() - t0
        log.info(f"   Hold-out: {{primary(test_m):.4f}}  gap={{gap:.4f}}  score={{score:.4f}}  {{elapsed:.1f}}s")

        results.append({{
            "model":         name,
            "family":        family,
            "status":        "ok",
            "best_params":   best_params,
            "train":         train_m,
            "test":          test_m,
            "cv_test_mean":  cv_mean,
            "cv_test_std":   cv_std,
            "cv_train_mean": cv_tr,
            "gap":           gap,
            "underfit":      underfit,
            "score":         float(score),
            "elapsed_sec":   elapsed,
            "_pipe":         final_pipe,
        }})
    except Exception as e:
        log.error(f"   FAILED: {{e}}", exc_info=True)
        results.append({{"model": name, "family": family, "status": "error", "error": str(e)}})

# â”€â”€ Model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if task_type == "regression":
    run_model("LinearRegression", lambda p: LinearRegression(),
              use_scaling=True, family="linear")
    run_model("Ridge",
              lambda p: Ridge(alpha=p.get("alpha", 1.0), random_state=RANDOM_SEED),
              lambda t: {{"alpha": t.suggest_float("alpha", 1e-3, 50.0, log=True)}},
              use_scaling=True, family="linear")
    run_model("Lasso",
              lambda p: Lasso(alpha=p.get("alpha", 0.01), random_state=RANDOM_SEED,
                              max_iter=5000),
              lambda t: {{"alpha": t.suggest_float("alpha", 1e-5, 1.0, log=True)}},
              use_scaling=True, family="linear")
    run_model("RandomForestRegressor",
              lambda p: RandomForestRegressor(
                  n_estimators=p.get("n_estimators", 500),
                  max_depth=p.get("max_depth", None),
                  min_samples_leaf=p.get("min_samples_leaf", 2),
                  max_features=p.get("max_features", "sqrt"),
                  random_state=RANDOM_SEED, n_jobs=N_JOBS),
              lambda t: {{
                  "n_estimators":     t.suggest_int("n_estimators", 200, 800),
                  "max_depth":        t.suggest_int("max_depth", 3, 20),
                  "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 15),
                  "max_features":     t.suggest_categorical("max_features",
                                          ["sqrt", "log2", 0.5, 0.8]),
              }}, family="tree")
    run_model("GradientBoostingRegressor",
              lambda p: GradientBoostingRegressor(
                  n_estimators=p.get("n_estimators", 300),
                  learning_rate=p.get("learning_rate", 0.05),
                  max_depth=p.get("max_depth", 3),
                  subsample=p.get("subsample", 0.8),
                  min_samples_leaf=p.get("min_samples_leaf", 5),
                  random_state=RANDOM_SEED),
              lambda t: {{
                  "n_estimators":     t.suggest_int("n_estimators", 200, 800),
                  "learning_rate":    t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                  "max_depth":        t.suggest_int("max_depth", 2, 6),
                  "subsample":        t.suggest_float("subsample", 0.6, 1.0),
                  "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 20),
              }}, family="boosting")
else:
    run_model("LogisticRegression",
              lambda p: LogisticRegression(C=p.get("C", 1.0), max_iter=3000,
                                           random_state=RANDOM_SEED),
              lambda t: {{"C": t.suggest_float("C", 1e-3, 10.0, log=True)}},
              use_scaling=True, family="linear")
    run_model("RandomForestClassifier",
              lambda p: RandomForestClassifier(
                  n_estimators=p.get("n_estimators", 500),
                  max_depth=p.get("max_depth", None),
                  min_samples_leaf=p.get("min_samples_leaf", 2),
                  max_features=p.get("max_features", "sqrt"),
                  random_state=RANDOM_SEED, n_jobs=N_JOBS),
              lambda t: {{
                  "n_estimators":     t.suggest_int("n_estimators", 200, 800),
                  "max_depth":        t.suggest_int("max_depth", 3, 20),
                  "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 15),
                  "max_features":     t.suggest_categorical("max_features", ["sqrt", "log2"]),
              }}, family="tree")
    run_model("GradientBoostingClassifier",
              lambda p: GradientBoostingClassifier(
                  n_estimators=p.get("n_estimators", 300),
                  learning_rate=p.get("learning_rate", 0.05),
                  max_depth=p.get("max_depth", 3),
                  subsample=p.get("subsample", 0.8),
                  min_samples_leaf=p.get("min_samples_leaf", 5),
                  random_state=RANDOM_SEED),
              lambda t: {{
                  "n_estimators":     t.suggest_int("n_estimators", 200, 800),
                  "learning_rate":    t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                  "max_depth":        t.suggest_int("max_depth", 2, 6),
                  "subsample":        t.suggest_float("subsample", 0.6, 1.0),
                  "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 20),
              }}, family="boosting")

# â”€â”€ LightGBM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import lightgbm as lgb
    _lgb_kw = dict(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=-1)
    if task_type == "regression":
        run_model("LGBMRegressor",
                  lambda p: lgb.LGBMRegressor(
                      n_estimators=p.get("n_estimators", 400),
                      learning_rate=p.get("learning_rate", 0.05),
                      num_leaves=p.get("num_leaves", min(31, MAX_LEAVES)),
                      min_child_samples=p.get("min_child_samples", 20),
                      subsample=p.get("subsample", 0.8),
                      colsample_bytree=p.get("colsample_bytree", 0.8),
                      reg_alpha=p.get("reg_alpha", 0.1),
                      reg_lambda=p.get("reg_lambda", 1.0), **_lgb_kw),
                  lambda t: {{
                      "n_estimators":      t.suggest_int("n_estimators", 100, 800),
                      "learning_rate":     t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "num_leaves":        t.suggest_int("num_leaves", 8, MAX_LEAVES),
                      "min_child_samples": t.suggest_int("min_child_samples", 10, 100),
                      "subsample":         t.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree":  t.suggest_float("colsample_bytree", 0.5, 1.0),
                      "reg_alpha":         t.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                      "reg_lambda":        t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                  }}, family="boosting")
    else:
        run_model("LGBMClassifier",
                  lambda p: lgb.LGBMClassifier(
                      n_estimators=p.get("n_estimators", 400),
                      learning_rate=p.get("learning_rate", 0.05),
                      num_leaves=p.get("num_leaves", min(31, MAX_LEAVES)),
                      min_child_samples=p.get("min_child_samples", 20),
                      subsample=p.get("subsample", 0.8),
                      colsample_bytree=p.get("colsample_bytree", 0.8),
                      reg_alpha=p.get("reg_alpha", 0.1),
                      reg_lambda=p.get("reg_lambda", 1.0), **_lgb_kw),
                  lambda t: {{
                      "n_estimators":      t.suggest_int("n_estimators", 100, 800),
                      "learning_rate":     t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "num_leaves":        t.suggest_int("num_leaves", 8, MAX_LEAVES),
                      "min_child_samples": t.suggest_int("min_child_samples", 10, 100),
                      "subsample":         t.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree":  t.suggest_float("colsample_bytree", 0.5, 1.0),
                      "reg_alpha":         t.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                      "reg_lambda":        t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                  }}, family="boosting")
    log.info("LightGBM added.")
except ImportError:
    log.warning("LightGBM not installed â€” skipping.")

# â”€â”€ XGBoost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import xgboost as xgb
    _xgb_kw = dict(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0)
    if task_type == "regression":
        run_model("XGBRegressor",
                  lambda p: xgb.XGBRegressor(
                      n_estimators=p.get("n_estimators", 400),
                      learning_rate=p.get("learning_rate", 0.05),
                      max_depth=p.get("max_depth", 4),
                      min_child_weight=p.get("min_child_weight", 5),
                      subsample=p.get("subsample", 0.8),
                      colsample_bytree=p.get("colsample_bytree", 0.8),
                      reg_alpha=p.get("reg_alpha", 0.1),
                      reg_lambda=p.get("reg_lambda", 1.0),
                      eval_metric="rmse", **_xgb_kw),
                  lambda t: {{
                      "n_estimators":     t.suggest_int("n_estimators", 100, 800),
                      "learning_rate":    t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "max_depth":        t.suggest_int("max_depth", 2, 8),
                      "min_child_weight": t.suggest_int("min_child_weight", 3, 30),
                      "subsample":        t.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                      "reg_alpha":        t.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                      "reg_lambda":       t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                  }}, family="boosting")
    else:
        run_model("XGBClassifier",
                  lambda p: xgb.XGBClassifier(
                      n_estimators=p.get("n_estimators", 400),
                      learning_rate=p.get("learning_rate", 0.05),
                      max_depth=p.get("max_depth", 4),
                      min_child_weight=p.get("min_child_weight", 5),
                      subsample=p.get("subsample", 0.8),
                      colsample_bytree=p.get("colsample_bytree", 0.8),
                      reg_alpha=p.get("reg_alpha", 0.1),
                      reg_lambda=p.get("reg_lambda", 1.0),
                      eval_metric="logloss", **_xgb_kw),
                  lambda t: {{
                      "n_estimators":     t.suggest_int("n_estimators", 100, 800),
                      "learning_rate":    t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "max_depth":        t.suggest_int("max_depth", 2, 8),
                      "min_child_weight": t.suggest_int("min_child_weight", 3, 30),
                      "subsample":        t.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                      "reg_alpha":        t.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                      "reg_lambda":       t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                  }}, family="boosting")
    log.info("XGBoost added.")
except ImportError:
    log.warning("XGBoost not installed â€” skipping.")

# â”€â”€ CatBoost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    if task_type == "regression":
        run_model("CatBoostRegressor",
                  lambda p: CatBoostRegressor(
                      iterations=p.get("iterations", 600),
                      learning_rate=p.get("learning_rate", 0.05),
                      depth=p.get("depth", 5),
                      l2_leaf_reg=p.get("l2_leaf_reg", 5.0),
                      min_data_in_leaf=p.get("min_data_in_leaf", 10),
                      random_seed=RANDOM_SEED, verbose=False),
                  lambda t: {{
                      "iterations":       t.suggest_int("iterations", 200, 1000),
                      "learning_rate":    t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "depth":            t.suggest_int("depth", 3, 8),
                      "l2_leaf_reg":      t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                      "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 5, 50),
                  }}, family="boosting")
    else:
        run_model("CatBoostClassifier",
                  lambda p: CatBoostClassifier(
                      iterations=p.get("iterations", 600),
                      learning_rate=p.get("learning_rate", 0.05),
                      depth=p.get("depth", 5),
                      l2_leaf_reg=p.get("l2_leaf_reg", 5.0),
                      min_data_in_leaf=p.get("min_data_in_leaf", 10),
                      random_seed=RANDOM_SEED, verbose=False),
                  lambda t: {{
                      "iterations":       t.suggest_int("iterations", 200, 1000),
                      "learning_rate":    t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                      "depth":            t.suggest_int("depth", 3, 8),
                      "l2_leaf_reg":      t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                      "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 5, 50),
                  }}, family="boosting")
    log.info("CatBoost added.")
except ImportError:
    log.warning("CatBoost not installed â€” skipping.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  DIVERSE Stacking Ensemble â€” one best model per family
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ok_results = sorted(
    [r for r in results if r.get("status") == "ok" and r.get("score") is not None],
    key=lambda r: r["score"], reverse=True,
)

if ENABLE_STACKING and len(ok_results) >= 3:
    log.info("â–¶ Building DIVERSE stacking ensembleâ€¦")
    try:
        family_best: dict = {{}}
        for r in ok_results:
            fam = r.get("family", "other")
            if fam not in family_best:
                family_best[fam] = r

        stack_members = list(family_best.values())
        used = {{r["model"] for r in stack_members}}
        for r in ok_results:
            if len(stack_members) >= 3:
                break
            if r["model"] not in used:
                stack_members.append(r)
                used.add(r["model"])

        log.info(f"   Members: {{[r['model'] for r in stack_members]}}")
        estimators = [(r["model"], r["_pipe"]) for r in stack_members]
        scorer     = (make_scorer(r2_score) if task_type == "regression"
                      else make_scorer(f1_score, average="weighted"))

        if task_type == "regression":
            stack = StackingRegressor(estimators=estimators,
                                      final_estimator=Ridge(alpha=1.0),
                                      cv=5, n_jobs=N_JOBS)
        else:
            stack = StackingClassifier(estimators=estimators,
                                       final_estimator=LogisticRegression(
                                           max_iter=1000, random_state=RANDOM_SEED),
                                       cv=5, n_jobs=N_JOBS)

        stack.fit(X_train, y_train)
        test_m  = eval_metrics(y_test,  stack.predict(X_test))
        train_m = eval_metrics(y_train, stack.predict(X_train))
        sanity_check("StackingEnsemble", train_m, test_m, y_test.values)

        cv_res = cross_validate(clone(stack), X_train, y_train, cv=cv_splitter,
                                scoring=scorer, return_train_score=True, n_jobs=1)
        cv_m  = float(np.mean(cv_res["test_score"]))
        cv_sd = float(np.std(cv_res["test_score"]))
        score, gap, underfit, _ = composite_score(cv_m, cv_sd, train_m, test_m)
        log.info(f"   Stacking CV: {{cv_m:.4f}} Â± {{cv_sd:.4f}}  score={{score:.4f}}")

        stacking_entry = {{
            "model":        "StackingEnsemble",
            "family":       "ensemble",
            "status":       "ok",
            "best_params":  {{"base_models": [r["model"] for r in stack_members]}},
            "train":        train_m,
            "test":         test_m,
            "cv_test_mean": cv_m,
            "cv_test_std":  cv_sd,
            "gap":          gap,
            "underfit":     underfit,
            "score":        float(score),
            "_pipe":        stack,
        }}
        results.append(stacking_entry)
        ok_results.append(stacking_entry)
    except Exception as e:
        log.error(f"Stacking failed: {{e}}", exc_info=True)

ok_results.sort(key=lambda r: r["score"], reverse=True)
best = ok_results[0] if ok_results else None

if best:
    log.info(f"BEST: {{best['model']}} [{{best.get('family','')}}]  "
             f"score={{best['score']:.4f}}  cv={{best.get('cv_test_mean','?'):.4f}}")

# â”€â”€ Save trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _infer_project_root(path_str: str) -> Path:
    p = Path(path_str).resolve()
    for parent in p.parents:
        if parent.name == "shared":
            return parent.parent
    return Path.cwd()

project_root = _infer_project_root(DATA_PATH)
models_dir = project_root / "ml_engineer" / f"models_{{MODEL_TAG}}"
models_dir.mkdir(parents=True, exist_ok=True)

def _safe_name(name: str) -> str:
    if not name:
        return "model"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")

saved_models = []
for r in ok_results:
    if r.get("status") != "ok":
        continue
    pipe = r.get("_pipe")
    if pipe is None:
        continue
    name = _safe_name(r.get("model", "model"))
    out_path = models_dir / f"{{name}}.joblib"
    try:
        joblib.dump(pipe, out_path)
        saved_models.append(out_path.name)
    except Exception as e:
        log.warning(f"Model save failed for {{name}}: {{e}}")

log.info(f"Saved {{len(saved_models)}} model(s) to {{models_dir}}")

# â”€â”€ SHAP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
feature_importances = {{}}
if ENABLE_SHAP and best and "_pipe" in best:
    try:
        import shap
        pipe     = best["_pipe"]
        pre_step = pipe.named_steps.get("pre")
        model    = pipe.named_steps.get("model")
        if pre_step and model:
            X_test_t  = pre_step.transform(X_test)
            explainer = shap.Explainer(model, X_test_t)
            shap_vals = explainer(X_test_t, check_additivity=False)
            try:
                feat_names = pre_step.get_feature_names_out()
            except Exception:
                feat_names = np.array([f"f{{i}}" for i in range(X_test_t.shape[1])])
            mean_abs = np.abs(
                shap_vals.values if hasattr(shap_vals, "values") else shap_vals
            ).mean(axis=0)
            if mean_abs.ndim > 1:
                mean_abs = mean_abs.mean(axis=-1)
            feature_importances = dict(sorted(
                zip(feat_names.tolist(), mean_abs.tolist()),
                key=lambda x: x[1], reverse=True,
            ))
            log.info(f"SHAP top-10: {{list(feature_importances.keys())[:10]}}")
    except ImportError:
        log.warning("shap not installed â€” skipping.")
    except Exception as e:
        log.warning(f"SHAP failed: {{e}}")

# â”€â”€ Output payload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _clean(r):
    return {{k: v for k, v in r.items() if k != "_pipe"}}

payload = {{
    "run_timestamp":            datetime.utcnow().isoformat() + "Z",
    "task_type":                task_type,
    "target_col":               target_col,
    "n_train":                  int(len(X_train)),
    "n_test":                   int(len(X_test)),
    "n_cv_folds":               N_CV_FOLDS,
    "optuna_trials":            OPTUNA_TRIALS,
    "leaky_features_removed":   leaky_features,
    "suspect_features":         suspect_features,
    "feature_target_corr":      feature_corr,
    "results":                  [_clean(r) for r in results],
    "leaderboard":              [_clean(r) for r in ok_results[:10]],
    "best_model":               _clean(best) if best else None,
    "feature_importances_shap": feature_importances,
}}

print("__ML_RESULTS_BEGIN__")
print(json.dumps(payload, indent=2, default=str))
print("__ML_RESULTS_END__")

out_path = Path("ml_results.json")
out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
log.info(f"Results saved to {{out_path.resolve()}}")
log.info("Pipeline complete.")
'''

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

    def _find_output_txt(self) -> Path | None:
        repo_out = self._repo_root() / "shared" / "output.txt"
        if workspace.is_initialized and workspace.project_root:
            ws_out = Path(workspace.project_root) / "shared" / "output.txt"
            if ws_out.exists():
                return ws_out
        if repo_out.exists():
            return repo_out
        return None

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

N_JOBS = int(os.getenv("ML_N_JOBS", "1"))

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
for col in cat_cols:
    try:
        X[col] = X[col].astype("string")
    except Exception:
        X[col] = X[col].astype(str)

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
        n_jobs=N_JOBS,
    )
else:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=N_JOBS,
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
        auth_token = (payload.get("auth_token") or "").strip()
        worker_project_path = (payload.get("worker_project_path") or "").strip()
        target_col = (payload.get("target_col") or "").strip() or None

        marker = "__PROBE_OUTPUT_BEGIN__"
        if marker in content:
            probe_out = content.split(marker, 1)[1].strip()
            if workspace.is_initialized:
                workspace.write("ml_engineer", "probe_output.log", probe_out, task_id)
            await self.report(probe_out, task_id)
            return None

        if ("deploy" in c and ("local" in c or "locally" in c)) or "deploy locally" in c or "deploy local" in c:
            if not auth_token:
                await self.report("Local deploy requires an authenticated session. Please log in again.", task_id)
                return None
            deploy_cmd = (os.getenv("LOCAL_DEPLOY_CMD") or "").strip()
            deploy_port = (os.getenv("LOCAL_DEPLOY_PORT") or "8001").strip()
            deploy_cwd = (os.getenv("LOCAL_DEPLOY_CWD") or "").strip()
            if not deploy_cwd and worker_project_path:
                deploy_cwd = worker_project_path
            if not deploy_cwd and workspace.project_root:
                deploy_cwd = str(workspace.project_root)
            if not deploy_cwd:
                await self.report("Local deploy failed: no project directory provided.", task_id)
                return None

            # Resolve model and dataset tag from the user's request.
            ds_path, inferred_label = self._find_dataset_from_message(c)
            dataset_tag = "engineered" if "engineered" in c else ("raw" if "raw" in c else inferred_label)
            payload_results = self._load_ml_results(dataset_tag or "engineered") or self._load_ml_results("raw")
            model_hint = None
            if payload_results:
                model_result = self._match_model_in_results(c, payload_results) or payload_results.get("best_model")
                if model_result:
                    model_hint = model_result.get("model")
            if not model_hint:
                model_hint = self._extract_model_hint(c)

            base_dir = Path(deploy_cwd).resolve()
            model_path = self._resolve_local_model_path(base_dir, model_hint, dataset_tag)
            if not model_path:
                await self.report("Local deploy failed: could not find a trained model in the project folder.", task_id)
                return None

            task_type = payload_results.get("task_type") if payload_results else None
            target_col = payload_results.get("target_col") if payload_results else None
            model_arg = str(model_path)
            dataset_arg = str(ds_path) if ds_path else ""

            if not deploy_cmd:
                await self._ensure_local_serve_script(auth_token, str(base_dir))
                deploy_cmd = (
                    f"python ml_engineer/serve_model.py --model \"{model_arg}\" "
                    + (f"--dataset \"{dataset_arg}\" " if dataset_arg else "")
                    + f"--host 0.0.0.0 --port {deploy_port}"
                    + (f" --task {task_type}" if task_type else "")
                    + (f" --target {target_col}" if target_col else "")
                )
            try:
                result = await self._local_worker_exec(auth_token, deploy_cmd, detach=True, cwd=str(base_dir))
            except Exception as exc:
                await self.report(f"Local deploy failed: {exc}", task_id)
                return None
            await self.report(
                "Local deploy started via worker.\n"
                f"Command: {deploy_cmd}\n"
                f"CWD: {base_dir}\n"
                f"Result: {result.get('result') or result}",
                task_id,
            )
            return None

        if any(k in c for k in ("training process", "training proceeded", "how training", "model transparency")):
            if not workspace.is_initialized or not workspace.project_root:
                await self.report("Workspace not initialized. Run training first.", task_id)
                return None

            ds_path, label = self._find_dataset_from_message(c)
            if not ds_path:
                await self.report("No dataset found for this request. Upload data and run training first.", task_id)
                return None

            payload_results = self._load_ml_results(label or "engineered")
            if not payload_results:
                await self.report(
                    "No training results found for the requested dataset. Run training first.",
                    task_id,
                )
                return None

            model_result = None
            # Try to match requested model
            model_result = self._match_model_in_results(c, payload_results)
            if not model_result:
                # Fallback to best model if not explicitly matched
                model_result = payload_results.get("best_model")
            if not model_result:
                await self.report("Could not resolve model from training results.", task_id)
                return None

            model_name = model_result.get("model")
            params = model_result.get("best_params", {}) or {}
            task_type = payload_results.get("task_type", "regression")
            target_col = payload_results.get("target_col")

            code = self._build_transparency_code(
                Path(ds_path), model_name, params, task_type, target_col
            )
            exec_result = await self.run_code_in_jupyter_kernel(code, timeout_s=1200)
            output = exec_result.get("output", "") or "[no output]"

            if workspace.is_initialized:
                tp_name = self._next_training_process_name()
                workspace.write("ml_engineer", tp_name, output, task_id)

            await self.report(
                "Training process generated via model_transparency.\n"
                f"Saved: ml_engineer/{tp_name}",
                task_id,
            )
            return None

        if any(k in c for k in ["model", "train", "eda", "dataset", "pipeline", "predict", "classification", "deploy"]):
            eng_path = self.find_latest_engineered_dataset()
            raw_path = self._find_latest_raw_dataset()
            if not raw_path and not eng_path:
                await self.report(
                    "No dataset found in shared/datasets. Upload a file first.",
                    task_id,
                )
                return None

            await self.report("ML Engineer multi-model Optuna training starting (engineered first, then raw).", task_id)

            raw_payload = None
            eng_payload = None
            if eng_path:
                eng_payload = await self._run_training_for_dataset(Path(eng_path), "engineered", task_id, target_col)
            else:
                await self.report("Engineered dataset not found. Proceeding with raw dataset only.", task_id)
            if raw_path:
                raw_payload = await self._run_training_for_dataset(Path(raw_path), "raw", task_id, target_col)
            else:
                await self.report("Raw dataset not found. Proceeding with engineered dataset only.", task_id)

            # Preserve legacy output names using engineered if available, else raw
            if workspace.is_initialized:
                pass

            if workspace.is_initialized:
                if eng_payload and eng_path:
                    report, final_report = self._build_ml_reports(Path(eng_path), eng_payload)
                    workspace.write("ml_engineer", "report.txt", report, task_id)
                    workspace.write("ml_engineer", "final_report.txt", final_report, task_id)
                elif raw_payload and raw_path:
                    report, final_report = self._build_ml_reports(Path(raw_path), raw_payload)
                    workspace.write("ml_engineer", "report.txt", report, task_id)
                    workspace.write("ml_engineer", "final_report.txt", final_report, task_id)

            comparison = self._build_comparison_report(raw_payload, eng_payload, raw_path, eng_path)
            if workspace.is_initialized:
                workspace.write("ml_engineer", "comparison.txt", comparison, task_id)

            await self.report(
                "ML Engineer completed multi-model Optuna training.\n"
                "Saved: ml_engineer/jupyter_output_raw.txt, ml_engineer/jupyter_output_engineered.txt, "
                "ml_engineer/report_raw.txt, ml_engineer/final_report_raw.txt, "
                "ml_engineer/report_engineered.txt, ml_engineer/final_report_engineered.txt, "
                "ml_engineer/comparison.txt, "
                "ml_engineer/models_raw/*.joblib, ml_engineer/models_engineered/*.joblib",
                task_id,
            )
            await self.message(
                "orchestrator",
                "ML Engineer completed Optuna training on raw + engineered datasets. Comparison report ready.",
                task_id,
            )
            return None

        await self.report("ML Engineer ready. Waiting for model-related instruction.", task_id)
        return None

    def get_tools(self):
        return []
