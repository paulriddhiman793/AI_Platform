"""
api/server.py — FastAPI + WebSocket server

Fix: listeners are started exactly once via a module-level flag.
Uvicorn's reload/startup can fire the @app.on_event("startup") multiple
times in some configs — this guard prevents duplicate listeners.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import uuid
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Set
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from api.message_bus import bus, send_to_agent
from api.auth import authenticate, create_user, ensure_default_user, DEFAULT_EMAIL
from tools.rag_store import build_hybrid_index_from_text
from tools.workspace import workspace


app = FastAPI(title="AI Engineering Platform")

origins_env = (os.getenv("FRONTEND_ORIGINS") or "").strip()
cors_origins = [o.strip() for o in origins_env.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure default user exists
try:
    ensure_default_user()
except Exception:
    pass


@app.post("/auth/register")
async def register(payload: dict):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    ok, msg = create_user(email, password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": "User created."}


@app.post("/auth/login")
async def login(payload: dict):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    if not authenticate(email, password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    _load_tokens()
    token = uuid.uuid4().hex
    _active_tokens[token] = email
    _save_tokens()
    return {"status": "ok", "token": token, "email": email}


@app.post("/auth/verify")
async def verify(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    return {"status": "ok", "email": email}


@app.post("/open_project")
async def open_project(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    if not workspace.is_initialized or not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    _assert_project_owner(email)
    path = workspace.project_root
    return {"status": "ok", "path": str(path)}


def _safe_rel_path(rel: str) -> Path:
    rel = (rel or "").strip().lstrip("/").lstrip("\\")
    if not rel or ".." in rel.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    base = workspace.project_root
    if not base:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    full = (base / rel).resolve()
    if base not in full.parents and full != base:
        raise HTTPException(status_code=400, detail="Path outside project.")
    return full


@app.post("/files")
async def list_files(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    if not workspace.is_initialized or not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    _assert_project_owner(email)
    base = workspace.project_root
    files = []
    for rel in workspace.list_files():
        full = (base / rel)
        try:
            stat = full.stat()
            files.append({
                "path": rel.replace("\\", "/"),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except Exception:
            continue
    return {"status": "ok", "project_root": str(base), "files": files}


@app.post("/file")
async def read_file(payload: dict):
    token = payload.get("auth_token")
    rel = payload.get("path")
    email = _require_auth_token(token)
    if not workspace.is_initialized or not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    _assert_project_owner(email)
    full = _safe_rel_path(rel)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    data = full.read_bytes()
    max_bytes = 1_000_000
    truncated = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    head = data[:2048]
    is_binary = b"\x00" in head
    ext = full.suffix.lower()
    if is_binary and ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        b64 = base64.b64encode(data).decode("utf-8")
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")
        return {
            "status": "ok",
            "path": str(rel),
            "binary": True,
            "mime": mime,
            "content_b64": b64,
            "truncated": truncated,
        }
    if is_binary:
        return {
            "status": "ok",
            "path": str(rel),
            "binary": True,
            "mime": "application/octet-stream",
            "content_b64": None,
            "truncated": truncated,
        }
    text = data.decode("utf-8", errors="replace")
    return {
        "status": "ok",
        "path": str(rel),
        "binary": False,
        "content": text,
        "truncated": truncated,
    }


@app.post("/project_zip")
async def download_project_zip(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    if not workspace.is_initialized or not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    _assert_project_owner(email)
    buf = io.BytesIO()
    base = workspace.project_root
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in workspace.list_files():
            full = base / rel
            if full.is_file():
                zf.write(full, arcname=rel.replace("\\", "/"))
    buf.seek(0)
    filename = f"{workspace.project_name or 'project'}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/projects")
async def list_projects(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    base = _ensure_platform_root()
    projects = []
    for p in base.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        owner = _get_project_owner(p)
        if not owner:
            owner = _maybe_claim_legacy_project(p, email)
        if not owner or owner != email.lower():
            continue
        info = p / ".project_info"
        name = p.name
        created = None
        if info.exists():
            try:
                lines = info.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines:
                    if line.lower().startswith("project:"):
                        name = line.split(":", 1)[1].strip()
                    if line.lower().startswith("created:"):
                        created = line.split(":", 1)[1].strip()
            except Exception:
                pass
        try:
            stat = p.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            mtime = None
        projects.append({
            "id": p.name,
            "name": name,
            "root": str(p),
            "owner": owner,
            "created": created,
            "modified": mtime,
        })
    projects.sort(key=lambda x: x.get("modified") or "", reverse=True)
    return {"status": "ok", "projects": projects}


@app.post("/projects/select")
async def select_project(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    project_id = (payload.get("project_id") or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required.")
    base = _ensure_platform_root()
    project_root = (base / project_id).resolve()
    if not project_root.exists():
        raise HTTPException(status_code=404, detail="Project not found.")
    owner = _get_project_owner(project_root)
    if not owner:
        owner = _maybe_claim_legacy_project(project_root, email)
    if owner and owner != email.lower():
        raise HTTPException(status_code=403, detail="Forbidden.")
    workspace.load_project(project_root)
    global _active_project_owner
    _active_project_owner = email.lower()
    return {
        "status": "ok",
        "project_name": workspace.project_name,
        "project_root": str(workspace.project_root),
    }

connected_clients: Set[WebSocket] = set()
_recent_user_messages: dict[tuple[str, str], float] = {}
_recent_bus_messages: dict[tuple, float] = {}
_active_tokens: dict[str, str] = {}
_tokens_loaded = False
_active_project_owner: str | None = None


def _ensure_platform_root() -> Path:
    root = os.getenv("PLATFORM_STORAGE_ROOT")
    if root:
        base = Path(root)
    else:
        base = Path(__file__).resolve().parent.parent / "platform_projects"
    base.mkdir(parents=True, exist_ok=True)
    if not workspace.output_path:
        workspace.configure(str(base))
    return base


def _tokens_path() -> Path:
    base = _ensure_platform_root()
    return base / ".auth_tokens.json"


def _load_tokens() -> None:
    global _tokens_loaded, _active_tokens
    if _tokens_loaded:
        return
    path = _tokens_path()
    if not path.exists():
        _tokens_loaded = True
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _active_tokens.update({str(k): str(v) for k, v in data.items()})
    except Exception:
        pass
    _tokens_loaded = True


def _save_tokens() -> None:
    try:
        path = _tokens_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_active_tokens, indent=2), encoding="utf-8")
    except Exception:
        pass


def _dedup_bus_message(key: tuple, window_s: float = 1.5) -> bool:
    """Return True if duplicate within window; otherwise record and return False."""
    now_ts = time.time()
    last_ts = _recent_bus_messages.get(key, 0.0)
    if now_ts - last_ts < window_s:
        return True
    _recent_bus_messages[key] = now_ts
    if len(_recent_bus_messages) > 500:
        # Drop an arbitrary old key to keep map small
        _recent_bus_messages.pop(next(iter(_recent_bus_messages)))
    return False

# ── Guard: listeners only ever start once ─────────────────────────────────────
_listeners_started = False


def _safe_dataset_name(name: str) -> str:
    raw = (name or "dataset.csv").strip()
    raw = raw.replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    return cleaned or "dataset.csv"


def _require_auth(msg: dict) -> str:
    return _require_auth_token(msg.get("auth_token"))


def _require_auth_token(token: str) -> str:
    token = (token or "").strip()
    if not _tokens_loaded:
        _load_tokens()
    email = _active_tokens.get(token)
    if not token or not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _get_project_owner(project_root: Path) -> str | None:
    owner_file = project_root / ".owner"
    if owner_file.exists():
        try:
            return owner_file.read_text(encoding="utf-8", errors="replace").strip().lower()
        except Exception:
            return None
    # Legacy: try to parse from .project_info
    info = project_root / ".project_info"
    if info.exists():
        try:
            for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("owner:"):
                    return line.split(":", 1)[1].strip().lower()
        except Exception:
            return None
    return None


def _maybe_claim_legacy_project(project_root: Path, email: str) -> str | None:
    """Claim legacy projects that predate ownership tracking."""
    owner = _get_project_owner(project_root)
    if owner:
        return owner
    if email.lower() != DEFAULT_EMAIL.lower():
        return None
    try:
        (project_root / ".owner").write_text(email.lower(), encoding="utf-8")
        info_path = project_root / ".project_info"
        if info_path.exists():
            info_text = info_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not any(line.lower().startswith("owner:") for line in info_text):
                info_text.append(f"Owner: {email.lower()}")
                info_path.write_text("\n".join(info_text) + "\n", encoding="utf-8")
        return email.lower()
    except Exception:
        return None


def _assert_project_owner(email: str) -> None:
    if not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    owner = _get_project_owner(workspace.project_root)
    if owner and owner != email.lower():
        raise HTTPException(status_code=403, detail="Forbidden.")


def _run_transparency_for_dataset_sync(dataset_path: Path, output_path: Path) -> None:
    code = (
        "import pandas as pd\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from model_transparency import run_pipeline_from_df, infer_target_column\n"
        f"df = pd.read_csv(r'''{str(dataset_path)}''')\n"
        "target_col, _, _, _ = infer_target_column(df, verbose=False)\n"
        "y = df[target_col]\n"
        "task_type = 'classification' if y.nunique(dropna=True) <= 20 else 'regression'\n"
        "model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=120, random_state=42)\n"
        "print('DATASET_PATH:', r'''%s''')\n" % str(dataset_path).replace("\\", "\\\\")
        + "print('DATASET_SHAPE:', df.shape)\n"
        "print('TARGET_COL:', target_col)\n"
        "print('TASK_TYPE:', task_type)\n"
        "run_pipeline_from_df(model=model, df=df, target_col=target_col, task_type=task_type, test_size=0.2, scale=(task_type=='regression'), cv=3, n_walkthrough=2)\n"
    )
    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        timeout=900,
        check=False,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        output = output + f"\n\n[runner] returncode={proc.returncode}\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")


def _run_phase3_checks_sync(project_root: Path) -> tuple[int, str]:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "phase3_checks.py"),
        "--project-root",
        str(project_root),
    ]
    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=1800,
        check=False,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


async def _phase3_check_background(project_root: Path, task_id: str | None) -> None:
    await broadcast_to_gui({
        "type": "team_message",
        "from": "server",
        "to": "team",
        "content": "Phase 3 checks started.",
        "tag": "STATUS",
        "task_id": task_id,
    })
    try:
        rc, output = await asyncio.to_thread(_run_phase3_checks_sync, project_root)
        try:
            log_path = project_root / "shared" / "phase3_check.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding="utf-8")
        except Exception:
            pass

        await broadcast_to_gui({
            "type": "team_message",
            "from": "server",
            "to": "team",
            "content": "Phase 3 checks completed." + (f" Return code: {rc}" if rc != 0 else ""),
            "tag": "DONE" if rc == 0 else "ALERT",
            "task_id": task_id,
        })

        if workspace.is_initialized and workspace.project_root:
            try:
                files = []
                for rel in workspace.list_files():
                    parts = Path(rel).parts
                    if not parts:
                        continue
                    agent_id = parts[0]
                    filename = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
                    files.append({
                        "agent_id": agent_id,
                        "filename": filename,
                        "full_path": str(workspace.project_root / rel),
                    })
                await broadcast_to_gui({
                    "type": "files_snapshot",
                    "from": "server",
                    "to": "gui",
                    "content": "files_snapshot",
                    "tag": None,
                    "task_id": None,
                    "extra": {"files": files},
                })
            except Exception:
                pass
    except Exception:
        await broadcast_to_gui({
            "type": "team_message",
            "from": "server",
            "to": "team",
            "content": "Phase 3 checks failed to run.",
            "tag": "ALERT",
            "task_id": task_id,
        })


async def _process_dataset_background(dataset_path: Path) -> None:
    if not workspace.is_initialized or not workspace.project_root:
        return
    shared_dir = workspace.project_root / "shared"
    out_path = shared_dir / "output.txt"
    rag_dir = shared_dir / "rag"

    try:
        await asyncio.to_thread(_run_transparency_for_dataset_sync, dataset_path, out_path)
        text = out_path.read_text(encoding="utf-8", errors="replace")
        await asyncio.to_thread(build_hybrid_index_from_text, text, rag_dir)
    except Exception:
        # Silent by design for GUI; backend logs only.
        pass


# ─── Broadcast to all GUI clients ─────────────────────────────────────────────

async def broadcast_to_gui(message: dict) -> None:
    if not connected_clients:
        return
    payload = json.dumps({**message, "timestamp": datetime.utcnow().isoformat()})
    dead = set()
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


# ─── Bus listeners (each runs exactly once) ───────────────────────────────────

async def _listen_orchestrator_inbox() -> None:
    q = bus.subscribe("orchestrator.inbox")
    while True:
        envelope = await q.get()
        p = envelope["payload"]
        if _dedup_bus_message(("team", p.get("from"), p.get("content"), p.get("task_id"))):
            continue
        await broadcast_to_gui({
            "type":    "team_message",
            "from":    p.get("from", "unknown"),
            "to":      "team",
            "content": p.get("content", ""),
            "tag":     _detect_tag(p.get("content", "")),
            "task_id": p.get("task_id"),
        })


async def _listen_p2p() -> None:
    q = bus.subscribe("p2p.monitor")
    while True:
        envelope = await q.get()
        p = envelope["payload"]
        if _dedup_bus_message(("p2p", p.get("from"), p.get("to"), p.get("content"), p.get("task_id"))):
            continue
        await broadcast_to_gui({
            "type":    "p2p_message",
            "from":    p.get("from"),
            "to":      p.get("to"),
            "content": p.get("content", ""),
            "tag":     None,
            "task_id": p.get("task_id"),
        })


async def _listen_user_output() -> None:
    q = bus.subscribe("user.output")
    while True:
        envelope = await q.get()
        p = envelope["payload"]
        await broadcast_to_gui({
            "type":    "team_message",
            "from":    p.get("from"),
            "to":      "user",
            "content": p.get("content", ""),
            "tag":     "DONE",
            "task_id": p.get("task_id"),
        })


async def _listen_file_events() -> None:
    q = bus.subscribe("workspace.files")
    while True:
        envelope = await q.get()
        p = envelope["payload"]
        await broadcast_to_gui({
            "type":    "file_written",
            "from":    p.get("agent_id"),
            "to":      "gui",
            "content": p.get("content", ""),
            "tag":     "STATUS",
            "task_id": p.get("task_id"),
            "extra": {
                "agent_id":  p.get("agent_id"),
                "filename":  p.get("filename"),
                "full_path": p.get("path"),
            },
        })


async def _listen_agent_status() -> None:
    q = bus.subscribe("agent.status")
    while True:
        envelope = await q.get()
        p = envelope["payload"]
        await broadcast_to_gui({
            "type":    "agent_status",
            "from":    p.get("agent_id"),
            "to":      "gui",
            "content": p.get("status"),
            "tag":     None,
            "task_id": None,
        })


def start_listeners_once() -> None:
    """Call this once at startup. Safe to call multiple times — guarded."""
    global _listeners_started
    if _listeners_started:
        print("[SERVER] Listeners already running — skipping duplicate start.")
        return
    _listeners_started = True
    asyncio.create_task(_listen_orchestrator_inbox())
    asyncio.create_task(_listen_p2p())
    asyncio.create_task(_listen_user_output())
    asyncio.create_task(_listen_file_events())
    asyncio.create_task(_listen_agent_status())
    print("[SERVER] Bus listeners started (once).")


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[SERVER] GUI connected. Clients: {len(connected_clients)}")

    # Send project info on connect
    if workspace.is_initialized:
        await websocket.send_text(json.dumps({
            "type":      "project_info",
            "from":      "server",
            "to":        "gui",
            "content":   f"Project: {workspace.project_name}",
            "tag":       "STATUS",
            "timestamp": datetime.utcnow().isoformat(),
            "task_id":   None,
            "extra": {
                "project_name": workspace.project_name,
                "project_root": str(workspace.project_root),
            },
        }))
        # Send a snapshot of existing files so the GUI can populate the Files panel.
        try:
            files = []
            for rel in workspace.list_files():
                parts = Path(rel).parts
                if not parts:
                    continue
                agent_id = parts[0]
                filename = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
                files.append({
                    "agent_id": agent_id,
                    "filename": filename,
                    "full_path": str(workspace.project_root / rel),
                })
            if files:
                await websocket.send_text(json.dumps({
                    "type":      "files_snapshot",
                    "from":      "server",
                    "to":        "gui",
                    "content":   "files_snapshot",
                    "tag":       None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_id":   None,
                    "extra": {
                        "files": files,
                    },
                }))
        except Exception:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # ── Ping ──────────────────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong", "from": "server", "to": "gui",
                    "content": "pong", "tag": None,
                    "timestamp": datetime.utcnow().isoformat(), "task_id": None,
                }))
                continue

            # ---- Initialize project from GUI ----
            if msg_type == "init_project":
                try:
                    email = _require_auth(msg)
                except HTTPException:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "Project init failed: unauthorized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue

                project_name = (msg.get("project_name") or "").strip()
                if not project_name:
                    project_name = f"{email.split('@')[0]}_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                try:
                    _ensure_platform_root()
                    project_root = workspace.new_project(project_name)
                    try:
                        (project_root / ".owner").write_text(email.lower(), encoding="utf-8")
                        info_path = project_root / ".project_info"
                        if info_path.exists():
                            info_text = info_path.read_text(encoding="utf-8", errors="replace").splitlines()
                            if not any(line.lower().startswith("owner:") for line in info_text):
                                info_text.append(f"Owner: {email.lower()}")
                                info_path.write_text("\n".join(info_text) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    global _active_project_owner
                    _active_project_owner = email.lower()

                    await broadcast_to_gui({
                        "type": "project_info",
                        "from": "server",
                        "to": "gui",
                        "content": f"Project: {workspace.project_name}",
                        "tag": "STATUS",
                        "timestamp": datetime.utcnow().isoformat(),
                        "task_id": msg.get("task_id"),
                        "extra": {
                            "project_name": workspace.project_name,
                            "project_root": str(project_root),
                        },
                    })
                    try:
                        files = []
                        for rel in workspace.list_files():
                            parts = Path(rel).parts
                            if not parts:
                                continue
                            agent_id = parts[0]
                            filename = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
                            files.append({
                                "agent_id": agent_id,
                                "filename": filename,
                                "full_path": str(workspace.project_root / rel),
                            })
                        await broadcast_to_gui({
                            "type": "files_snapshot",
                            "from": "server",
                            "to": "gui",
                            "content": "files_snapshot",
                            "tag": None,
                            "task_id": None,
                            "extra": {"files": files},
                        })
                    except Exception:
                        pass
                except Exception as e:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": f"Project init failed: {e}",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                continue

            # ---- Phase 3 checks ----
            if msg_type == "phase3_check":
                try:
                    _require_auth(msg)
                except HTTPException:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "Phase 3 check failed: unauthorized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue
                if not workspace.is_initialized or not workspace.project_root:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "Phase 3 check failed: project not initialized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue
                asyncio.create_task(_phase3_check_background(workspace.project_root, msg.get("task_id")))
                continue

            # ---- GitHub connect (store credentials/config only) ----
            if msg_type == "github_connect":
                try:
                    _require_auth(msg)
                except HTTPException:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "GitHub connect failed: unauthorized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue
                if not workspace.is_initialized or not workspace.project_root:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "GitHub connect failed: project not initialized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue
                token = (msg.get("token") or "").strip()
                owner = (msg.get("owner") or "").strip()
                repo = (msg.get("repo") or "").strip()
                visibility = (msg.get("visibility") or "private").strip().lower()
                if visibility not in ("private", "public"):
                    visibility = "private"
                if not token or not repo:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "GitHub connect failed: token and repo name are required.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue

                cfg = {
                    "token": token,
                    "owner": owner,
                    "repo": repo,
                    "visibility": visibility,
                }
                try:
                    cfg_path = workspace.project_root / "shared" / "github_config.json"
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "GitHub connected. Repo will be created and synced only when you ask the GitHub agent.",
                        "tag": "STATUS",
                        "task_id": msg.get("task_id"),
                    })
                except Exception as e:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": f"GitHub connect failed: {e}",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                continue

            # ── File write from frontend ───────────────────────────────────
            if msg_type == "file_write":
                try:
                    _require_auth(msg)
                except HTTPException:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "server",
                        "to": "team",
                        "content": "File write blocked: unauthorized.",
                        "tag": "ALERT",
                        "task_id": msg.get("task_id"),
                    })
                    continue
                agent_id     = msg.get("agent_id", "shared")
                filename     = msg.get("filename", "output.txt")
                file_content = msg.get("content", "")
                from_agent   = msg.get("from", agent_id)
                if workspace.is_initialized:
                    try:
                        written_path = workspace.write(agent_id, filename, file_content)
                        await broadcast_to_gui({
                            "type":    "file_written",
                            "from":    from_agent,
                            "to":      "gui",
                            "content": f"File written: {filename}",
                            "tag":     "STATUS",
                            "task_id": None,
                            "extra": {
                                "agent_id":  agent_id,
                                "filename":  filename,
                                "full_path": str(written_path),
                            },
                        })
                    except Exception as e:
                        print(f"[SERVER] File write error: {e}")
                continue

            # â”€â”€ Dataset upload from GUI (base64 over WebSocket) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if msg_type == "dataset_upload":
                try:
                    _require_auth(msg)
                except HTTPException:
                    continue
                if not workspace.is_initialized:
                    _ensure_platform_root()
                    workspace.new_project(f"chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

                filename = _safe_dataset_name(msg.get("filename", "dataset.csv"))
                payload_b64 = msg.get("content_b64", "")
                task_id = msg.get("task_id") or str(uuid.uuid4())[:8]

                try:
                    blob = base64.b64decode(payload_b64, validate=True)
                    datasets_dir = workspace.project_root / "shared" / "datasets"
                    datasets_dir.mkdir(parents=True, exist_ok=True)
                    dataset_path = datasets_dir / filename
                    dataset_path.write_bytes(blob)
                    asyncio.create_task(_process_dataset_background(dataset_path))
                    await broadcast_to_gui({
                        "type": "dataset_uploaded",
                        "from": "server",
                        "to": "gui",
                        "content": "dataset_uploaded",
                        "tag": "STATUS",
                        "task_id": task_id,
                        "extra": {
                            "filename": filename,
                            "path": str(dataset_path),
                        },
                    })
                except Exception:
                    pass
                continue

            # ── User message → orchestrator ────────────────────────────────
            content = msg.get("content", "").strip()
            to      = msg.get("to", "team")
            task_id = msg.get("task_id") or str(uuid.uuid4())[:8]

            if not content:
                continue

            try:
                _require_auth(msg)
            except HTTPException:
                await broadcast_to_gui({
                    "type": "team_message",
                    "from": "server",
                    "to": "team",
                    "content": "Message blocked: unauthorized.",
                    "tag": "ALERT",
                    "task_id": task_id,
                })
                continue

            # Guard against accidental duplicate sends from multiple GUI WS connections.
            dedup_key = (to, content)
            now_ts = time.time()
            last_ts = _recent_user_messages.get(dedup_key, 0.0)
            if now_ts - last_ts < 1.2:
                continue
            _recent_user_messages[dedup_key] = now_ts

            target = "orchestrator" if to == "team" else to
            await send_to_agent(
                from_agent="user",
                to_agent=target,
                content=content,
                task_id=task_id,
            )

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"[SERVER] GUI disconnected. Remaining: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        print(f"[SERVER] WebSocket error: {e}")


# ─── REST ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "project":      workspace.project_name,
        "project_root": str(workspace.project_root) if workspace.project_root else None,
        "clients":      len(connected_clients),
        "listeners":    _listeners_started,
    }


@app.get("/files")
async def list_files():
    if not workspace.is_initialized:
        return {"files": []}
    return {"files": workspace.list_files(), "root": str(workspace.project_root)}


@app.post("/predict")
async def predict(features: dict):
    """
    Live prediction endpoint — loads the model written by ML Engineer.
    Returns real prediction if model exists, mock response otherwise.
    """
    import json
    from pathlib import Path

    # Try to load the real model from workspace
    if workspace.is_initialized:
        model_candidates = [
            workspace.project_root / "ml_engineer" / "model" / "model.joblib",
            workspace.project_root / "shared"      / "model.joblib",
        ]
        for model_path in model_candidates:
            if model_path.exists():
                try:
                    import joblib, pandas as pd
                    model = joblib.load(model_path)
                    df    = pd.DataFrame([features])
                    return {
                        "prediction":  bool(model.predict(df)[0]),
                        "probability": round(float(model.predict_proba(df)[0][1]), 4),
                        "source":      "real_model",
                        "model_path":  str(model_path),
                    }
                except Exception as e:
                    pass  # Fall through to mock

    # Mock response when model isn't trained yet
    age      = features.get("account_age", 24)
    duration = features.get("session_duration", 5.2)
    prob     = round(min(0.95, max(0.05, (age * 0.02 + duration * 0.05))), 4)
    return {
        "prediction":  prob > 0.5,
        "probability": prob,
        "source":      "mock",
        "note":        "Model not yet trained — run the pipeline first",
    }


@app.get("/metrics")
async def metrics():
    """Model performance metrics — reads from workspace if available."""
    if workspace.is_initialized:
        # Try to read actual metrics logged by ML Engineer
        try:
            metrics_path = workspace.project_root / "shared" / "metrics.json"
            if metrics_path.exists():
                import json
                return json.loads(metrics_path.read_text())
        except Exception:
            pass
    return {
        "accuracy":         94.1,
        "f1":               0.89,
        "auc":              0.96,
        "drift_score":      0.03,
        "latency_p95":      118,
        "inference_volume": 8420,
        "source":           "mock",
    }


# ─── Helper ───────────────────────────────────────────────────────────────────

def _detect_tag(content: str) -> str:
    c = content.lower()
    if any(k in c for k in ["✅", "complete", "done", "deployed", "approved", "passed", "ci passed"]):
        return "DONE"
    if any(k in c for k in ["🔴", "critical", "exploit", "blocked", "❌"]):
        return "ALERT"
    if any(k in c for k in ["found", "result", "report", "eda", "accuracy", "score", "metric"]):
        return "REPORT"
    return "STATUS"
