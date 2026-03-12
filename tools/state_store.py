"""
Phase 4 state store (no LLMs).
Persists task state, run history, and artifact index under shared/state/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.workspace import workspace


def _state_root() -> Path:
    if workspace.is_initialized and workspace.project_root:
        root = workspace.project_root / "shared" / "state"
    else:
        root = Path(__file__).resolve().parent.parent / "shared" / "state"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _append_jsonl(filename: str, payload: dict[str, Any]) -> None:
    path = _state_root() / filename
    line = json.dumps(payload, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_run_event(agent_id: str, event: str, task_id: str | None, content: str | None, extra: dict | None = None) -> None:
    payload = {
        "ts": time.time(),
        "agent_id": agent_id,
        "event": event,
        "task_id": task_id,
        "content_preview": (content or "")[:200],
        "extra": extra or {},
    }
    _append_jsonl("runs.jsonl", payload)


def update_task_state(task_id: str | None, updates: dict[str, Any]) -> None:
    if not task_id:
        return
    root = _state_root()
    current_path = root / "tasks_current.json"
    if current_path.exists():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    else:
        current = {}
    task = current.get(task_id, {})
    task.update(updates)
    current[task_id] = task
    current_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    _append_jsonl("tasks.jsonl", {"ts": time.time(), "task_id": task_id, **updates})


def log_artifact(agent_id: str, filename: str, full_path: str, task_id: str | None, size_bytes: int | None = None) -> None:
    payload = {
        "ts": time.time(),
        "agent_id": agent_id,
        "filename": filename,
        "path": full_path,
        "task_id": task_id,
        "size_bytes": size_bytes,
    }
    _append_jsonl("artifacts.jsonl", payload)

