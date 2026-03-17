"""
Shared base class for all platform agents.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import importlib.util
import time
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from api.message_bus import bus, send_to_agent, broadcast, publish_status
from tools.rag_store import hybrid_search, build_hybrid_index_from_texts
from tools.workspace import workspace


class BaseAgent(ABC):
    AGENT_ID: str = "base"
    SYSTEM_PROMPT: str = "You are a helpful AI agent."
    _LLM_LOCK: asyncio.Lock | None = None
    _LLM_SEMAPHORE: asyncio.Semaphore | None = None
    _LLM_LAST_CALL_TS: float = 0.0
    _LLM_COOLDOWN_UNTIL: float = 0.0
    _GROQ_REQ_TIMES: list[float] = []
    _GROQ_TOKEN_WINDOW: list[tuple[float, int]] = []
    _GROQ_DAY: str | None = None
    _GROQ_DAY_COUNT: int = 0
    _GROQ_DAY_TOKENS: int = 0

    def __init__(self):
        self.inbox: asyncio.Queue = bus.subscribe(f"agent.{self.AGENT_ID}")
        self._memory: dict[str, list[dict]] = {}
        self.status: str = "idle"
        self._seen: set[str] = set()

    async def run(self) -> None:
        print(f"[{self.AGENT_ID}] Online and listening on agent.{self.AGENT_ID}")
        while True:
            envelope = await self.inbox.get()
            asyncio.create_task(self._dispatch(envelope))

    async def _dispatch(self, envelope: dict) -> None:
        payload = envelope["payload"]
        dedup_key = f"{payload.get('task_id','')}:{payload.get('content','')}"
        if dedup_key in self._seen:
            print(f"[{self.AGENT_ID}] Duplicate message dropped: {dedup_key[:60]}")
            return
        self._seen.add(dedup_key)
        if len(self._seen) > 200:
            self._seen.pop()

        self.status = "working"
        await publish_status(self.AGENT_ID, "working")
        start_ts = time.time()
        try:
            from tools import state_store
            state_store.log_run_event(self.AGENT_ID, "task_start", payload.get("task_id"), payload.get("content"))
            state_store.update_task_state(payload.get("task_id"), {
                "agent_id": self.AGENT_ID,
                "status": "working",
                "started_at": start_ts,
                "content_preview": str(payload.get("content", ""))[:200],
            })
        except Exception:
            pass
        try:
            print(
                f"[{self.AGENT_ID}] Received task from {payload.get('from', '?')}: "
                f"{str(payload.get('content', ''))[:60]}..."
            )
            result = await self.handle_task(payload)
            if result:
                await self.report(result, task_id=payload.get("task_id"))
            try:
                from tools import state_store
                state_store.log_run_event(self.AGENT_ID, "task_complete", payload.get("task_id"), payload.get("content"), {
                    "duration_sec": round(time.time() - start_ts, 3),
                })
                state_store.update_task_state(payload.get("task_id"), {
                    "agent_id": self.AGENT_ID,
                    "status": "completed",
                    "ended_at": time.time(),
                    "duration_sec": round(time.time() - start_ts, 3),
                })
            except Exception:
                pass
        except Exception as exc:
            print(f"[{self.AGENT_ID}] ERROR in handle_task: {exc}")
            self.status = "error"
            await publish_status(self.AGENT_ID, "error")
            await self.report(
                f"Error in {self.AGENT_ID}: {str(exc)}. Please check logs or provide clarification.",
                task_id=envelope["payload"].get("task_id"),
            )
            try:
                from tools import state_store
                state_store.log_run_event(self.AGENT_ID, "task_error", payload.get("task_id"), payload.get("content"), {
                    "error": str(exc),
                    "duration_sec": round(time.time() - start_ts, 3),
                })
                state_store.update_task_state(payload.get("task_id"), {
                    "agent_id": self.AGENT_ID,
                    "status": "error",
                    "ended_at": time.time(),
                    "duration_sec": round(time.time() - start_ts, 3),
                    "error": str(exc),
                })
            except Exception:
                pass
        finally:
            if self.status != "error":
                self.status = "idle"
                await publish_status(self.AGENT_ID, "idle")

    @abstractmethod
    async def handle_task(self, payload: dict) -> str | None:
        ...

    def get_tools(self) -> list:
        return []

    async def message(self, to_agent: str, content: str, task_id: str = None, extra: dict | None = None) -> None:
        print(f"[{self.AGENT_ID}] -> [{to_agent}]: {content[:60]}...")
        await send_to_agent(
            from_agent=self.AGENT_ID,
            to_agent=to_agent,
            content=content,
            task_id=task_id,
            extra=extra,
        )

    async def report(self, content: str, task_id: str = None) -> None:
        print(f"[{self.AGENT_ID}] REPORT: {content[:60]}...")
        await broadcast(from_agent=self.AGENT_ID, content=content, task_id=task_id)

    async def _call_llm(self, user_message: str, system_prompt: str = None, model_override: str = None) -> str:
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            await asyncio.sleep(0.05)
            return f"[LLM unavailable] {self.AGENT_ID}: GROQ_API_KEY missing"

        model = (model_override or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()
        payload = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": self._groq_max_tokens(),
            "messages": [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        return await self._call_groq_with_retry(payload, api_key)

    async def _call_llm_with_history(self, history: list[dict], model_override: str = None) -> str:
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            await asyncio.sleep(0.05)
            last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            return f"[LLM unavailable] {self.AGENT_ID}: {last_user[:160]}"

        model = (model_override or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()
        payload = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": self._groq_max_tokens(),
            "messages": history,
        }
        return await self._call_groq_with_retry(payload, api_key)

    @classmethod
    def _llm_lock(cls) -> asyncio.Lock:
        if cls._LLM_LOCK is None:
            cls._LLM_LOCK = asyncio.Lock()
        return cls._LLM_LOCK

    @classmethod
    def _llm_semaphore(cls) -> asyncio.Semaphore:
        if cls._LLM_SEMAPHORE is None:
            raw = (os.getenv("GROQ_MAX_CONCURRENCY") or "1").strip()
            try:
                val = max(1, min(8, int(raw)))
            except Exception:
                val = 1
            cls._LLM_SEMAPHORE = asyncio.Semaphore(val)
        return cls._LLM_SEMAPHORE

    @staticmethod
    def _groq_max_tokens() -> int:
        raw = (os.getenv("GROQ_MAX_TOKENS") or "1200").strip()
        try:
            val = int(raw)
            return max(256, min(3200, val))
        except Exception:
            return 1200

    @staticmethod
    def _estimate_tokens_from_payload(payload: dict) -> int:
        try:
            messages = payload.get("messages", [])
            chars = sum(len(m.get("content", "")) for m in messages)
            in_tokens = max(1, int(chars / 4))
            out_tokens = int(payload.get("max_tokens") or 0)
            return max(1, in_tokens + out_tokens)
        except Exception:
            return max(1, int(payload.get("max_tokens") or 256))

    @staticmethod
    def _groq_limits(model: str) -> tuple[int, int, int, int]:
        # Env overrides. If unset, use model defaults for gpt-oss-120b.
        def _read(name: str) -> int:
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return 0
            try:
                return max(0, int(raw))
            except Exception:
                return 0

        rpm = _read("GROQ_RPM")
        rpd = _read("GROQ_RPD")
        tpm = _read("GROQ_TPM")
        tpd = _read("GROQ_TPD")

        if not any((rpm, rpd, tpm, tpd)) and model.strip().lower() == "openai/gpt-oss-120b":
            rpm = 30
            rpd = 1000
            tpm = 8000
            tpd = 200000

        return rpm, rpd, tpm, tpd

    async def _apply_groq_rate_limit_locked(self, tokens: int, model: str) -> str | None:
        rpm, rpd, tpm, tpd = self._groq_limits(model)
        if not any((rpm, rpd, tpm, tpd)):
            return None

        day = datetime.utcnow().strftime("%Y-%m-%d")
        if BaseAgent._GROQ_DAY != day:
            BaseAgent._GROQ_DAY = day
            BaseAgent._GROQ_DAY_COUNT = 0
            BaseAgent._GROQ_DAY_TOKENS = 0

        if rpd and BaseAgent._GROQ_DAY_COUNT >= rpd:
            return f"RPD limit reached ({rpd}/day)"
        if tpd and (BaseAgent._GROQ_DAY_TOKENS + tokens) > tpd:
            return f"TPD limit reached ({tpd}/day)"

        def _purge(now: float) -> None:
            BaseAgent._GROQ_REQ_TIMES[:] = [t for t in BaseAgent._GROQ_REQ_TIMES if now - t < 60]
            BaseAgent._GROQ_TOKEN_WINDOW[:] = [(t, tok) for (t, tok) in BaseAgent._GROQ_TOKEN_WINDOW if now - t < 60]

        now = time.monotonic()
        _purge(now)

        if rpm and len(BaseAgent._GROQ_REQ_TIMES) >= rpm:
            wait = 60 - (now - BaseAgent._GROQ_REQ_TIMES[0])
            if wait > 0:
                await asyncio.sleep(wait)
            now = time.monotonic()
            _purge(now)

        if tpm:
            token_sum = sum(tok for _, tok in BaseAgent._GROQ_TOKEN_WINDOW)
            if token_sum + tokens > tpm and BaseAgent._GROQ_TOKEN_WINDOW:
                wait = 60 - (now - BaseAgent._GROQ_TOKEN_WINDOW[0][0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.monotonic()
                _purge(now)

        now2 = time.monotonic()
        BaseAgent._GROQ_REQ_TIMES.append(now2)
        BaseAgent._GROQ_TOKEN_WINDOW.append((now2, tokens))
        BaseAgent._GROQ_DAY_COUNT += 1
        BaseAgent._GROQ_DAY_TOKENS += tokens
        return None

    @staticmethod
    def _sync_groq_call(payload: dict, api_key: str) -> str:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if not content:
            try:
                preview = json.dumps(data)[:800]
            except Exception:
                preview = str(data)[:800]
            return f"[LLM empty] raw_response={preview}"
        return content

    async def _call_groq_with_retry(self, payload: dict, api_key: str) -> str:
        max_retries_raw = (os.getenv("GROQ_MAX_RETRIES") or "4").strip()
        min_interval_raw = (os.getenv("GROQ_MIN_INTERVAL_SEC") or "1.2").strip()
        try:
            max_retries = max(0, int(max_retries_raw))
        except Exception:
            max_retries = 4
        try:
            min_interval = max(0.0, min(10.0, float(min_interval_raw)))
        except Exception:
            min_interval = 1.2

        tokens_est = self._estimate_tokens_from_payload(payload)
        model_name = str(payload.get("model") or "").strip()
        for attempt in range(max_retries + 1):
            try:
                async with self._llm_semaphore():
                    async with self._llm_lock():
                        now_cd = time.monotonic()
                        if BaseAgent._LLM_COOLDOWN_UNTIL > now_cd:
                            await asyncio.sleep(BaseAgent._LLM_COOLDOWN_UNTIL - now_cd)
                        rate_err = await self._apply_groq_rate_limit_locked(tokens_est, model_name)
                        if rate_err:
                            return f"[LLM error] {rate_err}"
                        now = time.monotonic()
                        wait_s = max(0.0, min_interval - (now - BaseAgent._LLM_LAST_CALL_TS))
                        if wait_s > 0:
                            await asyncio.sleep(wait_s)
                        result = await asyncio.to_thread(self._sync_groq_call, payload, api_key)
                        BaseAgent._LLM_LAST_CALL_TS = time.monotonic()
                        return result
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429 and attempt < max_retries:
                    retry_after = (exc.response.headers.get("Retry-After", "") if exc.response else "").strip()
                    try:
                        delay = float(retry_after) if retry_after else (2 ** attempt)
                    except Exception:
                        delay = 2 ** attempt
                    cooldown = max(30.0, min(120.0, delay if delay else 60.0))
                    BaseAgent._LLM_COOLDOWN_UNTIL = time.monotonic() + cooldown
                    await asyncio.sleep(cooldown)
                    continue
                if status == 429:
                    return f"[LLM error] HTTPError 429: rate limit exceeded after retries ({max_retries + 1} attempts)"
                return f"[LLM error] HTTPError {status}: {exc}"
            except requests.RequestException as exc:
                if attempt < max_retries:
                    await asyncio.sleep(min(10.0, 1.2 * (attempt + 1)))
                    continue
                return f"[LLM error] RequestException: {exc}"
            except Exception as exc:
                return f"[LLM error] {type(exc).__name__}: {exc}"

        return "[LLM error] retry loop exhausted"

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def find_latest_uploaded_dataset(self) -> Path | None:
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
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def find_latest_engineered_dataset(self) -> Path | None:
        if not workspace.is_initialized or not workspace.project_root:
            return None
        ds_dir = workspace.project_root / "shared" / "datasets"
        if not ds_dir.exists():
            return None
        candidates = list(ds_dir.glob("*_engineered.csv"))
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    def rag_query_transparency_output(self, query: str, top_k: int = 6) -> list[dict]:
        if not workspace.is_initialized or not workspace.project_root:
            return []
        rag_dir = workspace.project_root / "shared" / "rag"
        if not rag_dir.exists():
            return []
        try:
            return hybrid_search(rag_dir, query=query, top_k=top_k)
        except Exception:
            return []

    def rag_query_reports(self, query: str, top_k: int = 6) -> list[dict]:
        if not workspace.is_initialized or not workspace.project_root:
            return []
        rag_dir = workspace.project_root / "shared" / "rag_reports"
        if not rag_dir.exists():
            return []
        try:
            return hybrid_search(rag_dir, query=query, top_k=top_k)
        except Exception:
            return []

    def build_reports_rag_index(self) -> bool:
        if not workspace.is_initialized or not workspace.project_root:
            return False
        root = workspace.project_root
        report_paths = [
            root / "data_scientist" / "report.md",
            root / "data_analyst" / "report.md",
            root / "data_analyst" / "business_report.md",
        ]
        texts = []
        for p in report_paths:
            if p.exists():
                try:
                    texts.append(f"FILE: {p.name}\n" + p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
        if not texts:
            return False
        rag_dir = root / "shared" / "rag_reports"
        try:
            build_hybrid_index_from_texts(texts, rag_dir)
            return True
        except Exception:
            return False

    async def _ensure_jupyter_kernel_stack(self) -> bool:
        has_client = importlib.util.find_spec("jupyter_client") is not None
        has_kernel = importlib.util.find_spec("ipykernel") is not None
        if has_client and has_kernel:
            return True

        def _install() -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "jupyter_client", "ipykernel"],
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )

        try:
            proc = await asyncio.to_thread(_install)
            if proc.returncode != 0:
                return False
        except Exception:
            return False

        return (
            importlib.util.find_spec("jupyter_client") is not None
            and importlib.util.find_spec("ipykernel") is not None
        )

    async def run_code_in_jupyter_kernel(self, code: str, timeout_s: int = 240) -> dict:
        """
        Execute Python code through a Jupyter kernel and return full output.
        Falls back to direct python execution if kernel stack is unavailable.
        """
        ready = await self._ensure_jupyter_kernel_stack()
        if not ready:
            return await self._run_python_code_locally(code, timeout_s=timeout_s, via="python-fallback")

        root = self._repo_root()
        prelude = (
            "import os, sys\n"
            f"os.chdir(r\"{str(root)}\")\n"
            f"_r = r\"{str(root)}\"\n"
            "sys.path.insert(0, _r) if _r not in sys.path else None\n"
        )
        code_to_run = prelude + "\n" + code

        def _run_kernel() -> dict:
            from jupyter_client import KernelManager

            km = KernelManager()
            kc = None
            out_chunks = []
            ok = True
            try:
                km.start_kernel()
                kc = km.client()
                kc.start_channels()
                kc.wait_for_ready(timeout=timeout_s)
                msg_id = kc.execute(code_to_run)
                while True:
                    msg = kc.get_iopub_msg(timeout=timeout_s)
                    if msg.get("parent_header", {}).get("msg_id") != msg_id:
                        continue
                    t = msg.get("msg_type", "")
                    c = msg.get("content", {})
                    if t == "stream":
                        out_chunks.append(c.get("text", ""))
                    elif t in ("execute_result", "display_data"):
                        txt = c.get("data", {}).get("text/plain", "")
                        if txt:
                            out_chunks.append(txt + "\n")
                    elif t == "error":
                        ok = False
                        out_chunks.append("\n".join(c.get("traceback", [])) + "\n")
                    elif t == "status" and c.get("execution_state") == "idle":
                        break
            except Exception as exc:
                ok = False
                out_chunks.append(f"Jupyter kernel execution failed: {exc}\n")
            finally:
                try:
                    if kc is not None:
                        kc.stop_channels()
                except Exception:
                    pass
                try:
                    km.shutdown_kernel(now=True)
                except Exception:
                    pass

            return {
                "success": ok,
                "output": "".join(out_chunks).strip(),
                "runner": "jupyter-kernel",
            }

        return await asyncio.to_thread(_run_kernel)

    async def _run_python_code_locally(self, code: str, timeout_s: int = 240, via: str = "python") -> dict:
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix=f"{self.AGENT_ID}_nb_",
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(code)
                tmp_file = Path(fh.name)

            def _run() -> subprocess.CompletedProcess:
                return subprocess.run(
                    [sys.executable, str(tmp_file)],
                    cwd=str(self._repo_root()),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_s,
                )

            proc = await asyncio.to_thread(_run)
            merged = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
            return {"success": proc.returncode == 0, "output": merged, "runner": via}
        except Exception as exc:
            return {"success": False, "output": f"Local execution failed: {exc}", "runner": via}
        finally:
            if tmp_file and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    async def _run_python_in_powershell(
        self,
        script_rel_path: str,
        timeout_s: int = 240,
    ) -> dict:
        """
        Run a Python script through PowerShell and capture full combined output.
        """
        root = self._repo_root()
        py = str(Path(sys.executable).resolve())
        script = str((root / script_rel_path).resolve())
        cmd = (
            '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
            '$env:PYTHONUTF8 = "1"; '
            '$env:PYTHONIOENCODING = "utf-8"; '
            '$env:LC_ALL = "C.UTF-8"; '
            f'& "{py}" "{script}" 2>&1 | Out-String'
        )

        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    cmd,
                ],
                cwd=str(root),
                capture_output=True,
                text=False,
                timeout=timeout_s,
                check=False,
            )

        try:
            proc = await asyncio.to_thread(_run)
            stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
            stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
            merged = "\n".join([stdout_text, stderr_text]).strip()
            return {
                "success": proc.returncode == 0,
                "stdout": merged,
                "stderr": "",
                "returncode": proc.returncode,
            }
        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"powershell runner failed: {exc}",
                "returncode": 1,
            }

    async def _run_python_in_docker(
        self,
        script_rel_path: str,
        timeout_s: int = 180,
        extra_env: dict | None = None,
    ) -> dict:
        root = self._repo_root()
        image = (os.getenv("AGENT_DOCKER_IMAGE") or "python:3.11-slim").strip()
        volume = f"{str(root)}:/workspace"
        pkg_list = (os.getenv("AGENT_DOCKER_PY_PKGS") or "numpy pandas scikit-learn").strip()
        run_script = (
            "python -m pip install -q --disable-pip-version-check "
            f"{pkg_list} && python {script_rel_path}"
        )

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            volume,
            "-w",
            "/workspace",
        ]
        if extra_env:
            for k, v in extra_env.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([image, "sh", "-lc", run_script])

        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        try:
            proc = await asyncio.to_thread(_run)
            out = {
                "success": proc.returncode == 0,
                "stdout": (proc.stdout or "").strip(),
                "stderr": (proc.stderr or "").strip(),
                "returncode": proc.returncode,
            }
            if out["success"]:
                return out
            if self._looks_like_docker_daemon_down(out["stderr"]):
                return await self._run_python_locally(script_rel_path, timeout_s)
            return out
        except FileNotFoundError:
            return await self._run_python_locally(script_rel_path, timeout_s)
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"docker run timeout after {timeout_s}s",
                "returncode": 124,
            }

    @staticmethod
    def _looks_like_docker_daemon_down(stderr: str) -> bool:
        s = (stderr or "").lower()
        return any(
            k in s
            for k in [
                "failed to connect to the docker api",
                "dockerdesktoplinuxengine",
                "the system cannot find the file specified",
                "cannot connect to the docker daemon",
                "error during connect",
            ]
        )

    async def _run_python_locally(self, script_rel_path: str, timeout_s: int) -> dict:
        """
        Windows-safe fallback when Docker isn't available/running.
        """
        root = self._repo_root()
        cmd = [sys.executable, script_rel_path]

        def _run_local() -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        try:
            proc = await asyncio.to_thread(_run_local)
            return {
                "success": proc.returncode == 0,
                "stdout": (proc.stdout or "").strip(),
                "stderr": (proc.stderr or "").strip() + "\n[probe executed locally: docker unavailable]",
                "returncode": proc.returncode,
            }
        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"local probe execution failed: {exc}",
                "returncode": 1,
            }

    async def run_model_transparency_probe(self, task_id: str | None = None) -> dict:
        """
        Run transparency_quick_probe.py and return raw output.
        """
        script_name = "transparency_quick_probe.py"
        result = await self._run_python_in_powershell(script_name, timeout_s=240)
        combined = "\n".join([result.get("stdout", ""), result.get("stderr", "")]).strip()
        return {
            "success": result.get("success", False),
            "script": script_name,
            "output": combined,
            "summary": "",
        }

    def _extract_code_block(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    async def _run_with_healing(
        self,
        code: str,
        max_retries: int = 5,
        task_id: str = None,
    ) -> dict:
        for attempt in range(1, max_retries + 1):
            result = await self._execute_in_sandbox(code)

            if result["success"]:
                print(f"[{self.AGENT_ID}] Code succeeded on attempt {attempt}")
                return {"success": True, "output": result["output"], "attempts": attempt}

            print(f"[{self.AGENT_ID}] Attempt {attempt} failed: {result['error'][:60]}")
            await self.report(
                f"Attempt {attempt}/{max_retries} failed: {result['error'][:120]}. "
                "Diagnosing and applying fix...",
                task_id,
            )
            fix_prompt = (
                "This Python code failed syntax/runtime checks in Docker.\n\n"
                f"ERROR:\n{result['error']}\n\n"
                f"CODE:\n{code}\n\n"
                "Return only corrected Python code."
            )
            fixed = await self._call_llm(fix_prompt)
            code = self._extract_code_block(fixed) or code

        await self.report(
            f"Could not auto-fix after {max_retries} attempts. Escalating to user.",
            task_id,
        )
        return {"success": False, "output": None, "attempts": max_retries}

    async def _execute_in_sandbox(self, code: str) -> dict:
        """
        Execute code in Docker by compiling it (`py_compile`).
        Falls back to local compile only if Docker is unavailable.
        """
        root = self._repo_root()
        tmp_dir = root / ".agent_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix=f"{self.AGENT_ID}_",
                dir=tmp_dir,
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(code)
                tmp_file = Path(fh.name)

            rel = tmp_file.relative_to(root).as_posix()
            image = (os.getenv("AGENT_DOCKER_IMAGE") or "python:3.11-slim").strip()
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{str(root)}:/workspace",
                "-w",
                "/workspace",
                image,
                "python",
                "-m",
                "py_compile",
                rel,
            ]

            def _run_docker() -> subprocess.CompletedProcess:
                return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)

            try:
                proc = await asyncio.to_thread(_run_docker)
                if proc.returncode == 0:
                    return {"success": True, "output": "Docker py_compile passed", "error": None}
                err = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
                return {"success": False, "output": None, "error": err or "docker py_compile failed"}
            except FileNotFoundError:
                pass
            except subprocess.TimeoutExpired:
                return {"success": False, "output": None, "error": "docker py_compile timed out"}

            def _run_local() -> subprocess.CompletedProcess:
                return subprocess.run(
                    [sys.executable, "-m", "py_compile", str(tmp_file)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )

            local_proc = await asyncio.to_thread(_run_local)
            if local_proc.returncode == 0:
                return {
                    "success": True,
                    "output": "Local py_compile passed (docker unavailable)",
                    "error": None,
                }
            local_err = "\n".join([local_proc.stdout or "", local_proc.stderr or ""]).strip()
            return {
                "success": False,
                "output": None,
                "error": local_err or "local py_compile failed",
            }
        finally:
            if tmp_file and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def read_file(self, agent_id: str, filename: str) -> str | None:
        try:
            return workspace.read(agent_id, filename)
        except Exception:
            return None

    def list_workspace_files(self, agent_id: str = None) -> list[str]:
        try:
            return workspace.list_files(agent_id)
        except Exception:
            return []

    def workspace_context(self, *file_specs: tuple[str, str]) -> str:
        if not workspace.is_initialized:
            return "[workspace not initialized]"
        parts = []
        for agent_id, filename in file_specs:
            content = self.read_file(agent_id, filename)
            if content:
                parts.append(f"=== {agent_id}/{filename} ===\n{content}\n")
            else:
                parts.append(f"=== {agent_id}/{filename} ===\n[file not yet written]\n")
        return "\n".join(parts) if parts else "[no files found]"

    def remember(self, key: str, value: str, namespace: str = "default") -> None:
        if namespace not in self._memory:
            self._memory[namespace] = []
        self._memory[namespace].append(
            {"key": key, "value": value, "ts": datetime.utcnow().isoformat()}
        )

    def recall(self, key: str, namespace: str = "default") -> str | None:
        entries = self._memory.get(namespace, [])
        for entry in reversed(entries):
            if entry["key"] == key:
                return entry["value"]
        return None

    def recall_all(self, namespace: str = "default") -> list[dict]:
        return list(self._memory.get(namespace, []))
