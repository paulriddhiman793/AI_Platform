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
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import structlog

from api.config import settings
from api.message_bus import bus, send_to_agent
from api.auth import authenticate, create_user, ensure_default_user, DEFAULT_EMAIL
from tools.rag_store import build_hybrid_index_from_text
from tools.workspace import workspace
from api.schemas import (
    RegisterRequest, LoginRequest, TokenRequest, WorkerExecRequest,
    WorkerWriteFileRequest, ProjectSelectRequest, FileReadRequest,
    PredictRequest, encrypt_data, decrypt_data
)


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_env_file(Path(__file__).resolve().parent.parent / ".env")

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

app = FastAPI(title="AI Engineering Platform")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS & Security Headers Middleware
origins_env = (settings.frontend_origins or "").strip()
cors_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
for local_origin in ("http://localhost:5173", "http://localhost:3000",
                     "http://127.0.0.1:5173", "http://127.0.0.1:3000"):
    if local_origin not in cors_origins:
        cors_origins.append(local_origin)
if not cors_origins:
    cors_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self' http: https: ws: wss: data: blob: 'unsafe-inline' 'unsafe-eval';"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "Accept", "Origin"],
)

# Ensure default user exists
try:
    ensure_default_user()
except Exception:
    pass

# Health endpoints
@app.get("/healthz")
@limiter.limit("100/minute")
async def health_check(request: Request) -> JSONResponse:
    """Liveness probe - returns 200 if server is running."""
    return JSONResponse({"status": "ok", "service": "ai-platform-api"})


@app.get("/readyz")
@limiter.limit("100/minute")
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe - checks dependencies (MongoDB, Redis)."""
    checks = {"mongodb": False, "redis": False, "workspace": False}

    # Check MongoDB
    try:
        from api.auth import _users_collection
        _users_collection().database.client.admin.command("ping")
        checks["mongodb"] = True
    except Exception:
        pass

    # Check Redis (for message bus)
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        checks["redis"] = True
    except Exception:
        pass

    # Check workspace
    checks["workspace"] = workspace.is_initialized

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    return JSONResponse(
        {"status": "ready" if all_ready else "not_ready", "checks": checks},
        status_code=status_code,
    )


@app.post("/auth/register")
@limiter.limit("5/minute")
async def register(request: Request, payload: RegisterRequest):
    ok, msg = create_user(payload.email, payload.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": "User created."}


@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest):
    if not authenticate(payload.email, payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    _load_tokens()
    token = uuid.uuid4().hex
    _active_tokens[token] = {"email": payload.email, "expires_at": time.time() + 86400}
    _save_tokens()
    return {"status": "ok", "token": token, "email": payload.email}


@app.post("/auth/verify")
@limiter.limit("60/minute")
async def verify(request: Request, payload: TokenRequest):
    email = _require_auth_token(payload.auth_token)
    return {"status": "ok", "email": email}


@app.post("/worker/pair")
@limiter.limit("20/minute")
async def worker_pair(request: Request, payload: TokenRequest):
    email = _require_auth_token(payload.auth_token)
    ttl_s = 900
    pair_token = _issue_pair_token(email, ttl_s=ttl_s)
    return {"status": "ok", "pair_token": pair_token, "expires_in": ttl_s}


@app.post("/worker/status")
@limiter.limit("60/minute")
async def worker_status(request: Request, payload: TokenRequest):
    email = _require_auth_token(payload.auth_token)
    ws = _worker_sessions.get(email.lower())
    return {"status": "ok", "connected": bool(ws)}


@app.get("/worker/download")
@limiter.limit("60/minute")
async def worker_download(request: Request):
    _require_auth_from_request(request)
    repo_root = Path(__file__).resolve().parent.parent
    worker_py = repo_root / "tools" / "local_worker.py"
    worker_req = repo_root / "tools" / "local_worker_requirements.txt"
    if not worker_py.exists() or not worker_req.exists():
        raise HTTPException(status_code=404, detail="Local worker package not found.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(worker_py, arcname="local_worker.py")
        zf.write(worker_req, arcname="local_worker_requirements.txt")
        zf.writestr(
            "README.txt",
            "Local ML Worker\n\n"
            "1) Install deps: pip install -r local_worker_requirements.txt\n"
            "2) Run: python local_worker.py --server <BACKEND_URL> --token <PAIR_TOKEN>\n",
        )
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="local_ml_worker.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/worker/exec")
@limiter.limit("20/minute")
async def worker_exec(request: Request, payload: WorkerExecRequest):
    email = _require_auth_token(payload.auth_token).lower()
    command = payload.command
    cwd = (payload.cwd or "").strip()
    detach = payload.detach
    if cwd:
        cwd = os.path.abspath(cwd)
        if not os.path.isdir(cwd):
            raise HTTPException(status_code=400, detail="cwd must be an existing directory.")
    timeout_s = payload.timeout_s
    ws = _worker_sessions.get(email)
    if not ws:
        raise HTTPException(status_code=409, detail="Local worker not connected.")
    job_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _worker_pending[job_id] = {"future": fut, "email": email}
    await ws.send_text(json.dumps({
        "type": "exec",
        "job_id": job_id,
        "command": command,
        "cwd": cwd,
        "detach": detach,
    }))
    try:
        result = await asyncio.wait_for(fut, timeout=float(timeout_s))
    except asyncio.TimeoutError:
        _worker_pending.pop(job_id, None)
        raise HTTPException(status_code=504, detail="Local worker timed out.")
    return {"status": "ok", "result": result}


@app.post("/worker/write_file")
@limiter.limit("20/minute")
async def worker_write_file(request: Request, payload: WorkerWriteFileRequest):
    email = _require_auth_token(payload.auth_token).lower()
    rel_path = payload.path
    content_b64 = payload.content_b64
    cwd = (payload.cwd or "").strip()
    timeout_s = payload.timeout_s
    try:
        raw_bytes = base64.b64decode(content_b64, validate=True)
        if len(raw_bytes) > 10_000_000:
            raise HTTPException(status_code=413, detail="File size exceeds 10MB limit.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 content.")
    ext = Path(rel_path).suffix.lower()
    if ext in {".exe", ".dll", ".so", ".sh", ".bat", ".cmd", ".msi"}:
        raise HTTPException(status_code=403, detail="Executable file extensions not allowed.")
    if cwd:
        cwd = os.path.abspath(cwd)
        if not os.path.isdir(cwd):
            raise HTTPException(status_code=400, detail="cwd must be an existing directory.")
    ws = _worker_sessions.get(email)
    if not ws:
        raise HTTPException(status_code=409, detail="Local worker not connected.")
    job_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _worker_pending[job_id] = {"future": fut, "email": email}
    await ws.send_text(json.dumps({
        "type": "write_file",
        "job_id": job_id,
        "path": rel_path,
        "content_b64": content_b64,
        "cwd": cwd,
    }))
    try:
        result = await asyncio.wait_for(fut, timeout=float(timeout_s))
    except asyncio.TimeoutError:
        _worker_pending.pop(job_id, None)
        raise HTTPException(status_code=504, detail="Local worker timed out.")
    return {"status": "ok", "result": result}


@app.post("/open_project")
async def open_project(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    if not workspace.is_initialized or not workspace.project_root:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    _assert_project_owner(email)
    path = workspace.project_root
    return {"status": "ok", "path": str(path)}


def _safe_rel_path(rel: str, base: Path = None) -> Path:
    raw_rel = (rel or "").strip()
    if not raw_rel:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not base:
        base = workspace.project_root
    if not base:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    base = base.resolve()
    try:
        candidate = Path(raw_rel)
        if candidate.is_absolute() or ":" in raw_rel[:3]:
            full = candidate.resolve()
            if base in full.parents or full == base:
                return full
    except Exception:
        pass
    cleaned = raw_rel.lstrip("/").lstrip("\\")
    if ".." in cleaned.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    full = (base / cleaned).resolve()
    if base not in full.parents and full != base:
        raise HTTPException(status_code=400, detail="Path outside project.")
    return full


@app.post("/files")
async def list_files(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    base = _resolve_authorized_project(payload, email)
    if not workspace.is_initialized or str(workspace.project_root.resolve()) != str(base):
        workspace.load_project(base)
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
@app.post("/file/read")
async def read_file(payload: dict):
    token = payload.get("auth_token")
    rel = payload.get("path")
    email = _require_auth_token(token)
    base = _resolve_authorized_project(payload, email)
    if not workspace.is_initialized or str(workspace.project_root.resolve()) != str(base):
        workspace.load_project(base)
    full = _safe_rel_path(rel, base)
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
            "is_binary": True,
            "mime": mime,
            "content": b64,
            "content_b64": b64,
            "truncated": truncated,
        }
    if is_binary:
        return {
            "status": "ok",
            "path": str(rel),
            "binary": True,
            "is_binary": True,
            "mime": "application/octet-stream",
            "content": None,
            "content_b64": None,
            "truncated": truncated,
        }
    text = data.decode("utf-8", errors="replace")
    return {
        "status": "ok",
        "path": str(rel),
        "binary": False,
        "is_binary": False,
        "content": text,
        "truncated": truncated,
    }


@app.get("/api/workspace/files/{agent_id}/{filename:path}")
@app.get("/workspace/files/{agent_id}/{filename:path}")
async def get_workspace_file(agent_id: str, filename: str, request: Request):
    proj_root_str = request.query_params.get("project_root", "").strip()
    base = None
    if proj_root_str and Path(proj_root_str).exists():
        base = Path(proj_root_str).resolve()
    elif workspace.is_initialized and workspace.project_root:
        base = workspace.project_root.resolve()
    if not base:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    target = base / agent_id / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)


@app.post("/project_zip")
async def download_project_zip(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    base = _resolve_authorized_project(payload, email)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for full in base.rglob("*"):
            if not full.is_file():
                continue
            rel = full.relative_to(base)
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                zf.write(full, arcname=str(rel).replace("\\", "/"))
            except Exception:
                continue
    zip_bytes = buf.getvalue()
    filename = f"{base.name or 'project'}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


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


def _assign_project_owner(project_root: Path, email: str) -> None:
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


@app.post("/projects/select")
async def select_project(payload: dict):
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    project_id = (payload.get("project_id") or payload.get("project_root") or payload.get("root") or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required.")
    base = _ensure_platform_root().resolve()
    candidate = Path(project_id)
    if not candidate.is_absolute():
        project_root = (base / project_id).resolve()
    else:
        project_root = candidate.resolve()
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
    settings = _read_project_settings(project_root)
    return {
        "status": "ok",
        "project_name": workspace.project_name,
        "project_root": str(workspace.project_root),
        "name": workspace.project_name,
        "root": str(workspace.project_root),
        "target_col": _normalize_target_col(settings.get("target_col")),
    }


@app.post("/projects/open")
async def open_or_create_project(payload: dict):
    global _active_project_owner
    token = payload.get("auth_token")
    email = _require_auth_token(token)
    name = (payload.get("name") or "").strip()
    root_raw = (payload.get("root") or payload.get("project_root") or payload.get("project_id") or "").strip()
    if not name and not root_raw:
        raise HTTPException(status_code=400, detail="Project name or root required.")

    base = _ensure_platform_root().resolve()
    target_root = None

    if root_raw:
        candidate = Path(root_raw)
        if not candidate.is_absolute():
            candidate = (base / root_raw).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            target_root = candidate

    if target_root:
        owner = _get_project_owner(target_root)
        if not owner:
            owner = _maybe_claim_legacy_project(target_root, email)
        if owner and owner != email.lower():
            raise HTTPException(status_code=403, detail="Forbidden: project owned by another user.")
        workspace.load_project(target_root)
        _active_project_owner = email.lower()
        settings = _read_project_settings(target_root)
        return {
            "status": "ok",
            "name": workspace.project_name,
            "root": str(workspace.project_root),
            "project_name": workspace.project_name,
            "project_root": str(workspace.project_root),
            "target_col": _normalize_target_col(settings.get("target_col")),
        }

    if not name:
        name = Path(root_raw).name if root_raw else "New Project"
    workspace.new_project(name)
    new_root = workspace.project_root
    _assign_project_owner(new_root, email)
    _active_project_owner = email.lower()
    settings = _read_project_settings(new_root)
    return {
        "status": "ok",
        "name": workspace.project_name,
        "root": str(workspace.project_root),
        "project_name": workspace.project_name,
        "project_root": str(workspace.project_root),
        "target_col": _normalize_target_col(settings.get("target_col")),
    }

connected_clients: Set[WebSocket] = set()
_ws_users: dict[WebSocket, str] = {}
_recent_user_messages: dict[tuple[str, str], float] = {}
_recent_bus_messages: dict[tuple, float] = {}
_active_tokens: dict[str, str] = {}
_tokens_loaded = False
_active_project_owner: str | None = None
_pair_tokens: dict[str, dict] = {}
_worker_sessions: dict[str, WebSocket] = {}
_worker_pending: dict[str, dict] = {}


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
        raw = path.read_text(encoding="utf-8")
        data = json.loads(decrypt_data(raw))
        if isinstance(data, dict):
            _active_tokens.update(data)
    except Exception:
        pass
    _tokens_loaded = True


def _save_tokens() -> None:
    try:
        path = _tokens_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encrypt_data(json.dumps(_active_tokens, indent=2)), encoding="utf-8")
    except Exception:
        pass


def _prune_pair_tokens(now_ts: float | None = None) -> None:
    now_ts = now_ts or time.time()
    expired = [k for k, v in _pair_tokens.items() if v.get("expires_at", 0) <= now_ts]
    for k in expired:
        _pair_tokens.pop(k, None)


def _issue_pair_token(email: str, ttl_s: int = 900) -> str:
    _prune_pair_tokens()
    token = uuid.uuid4().hex
    _pair_tokens[token] = {
        "email": email.lower(),
        "expires_at": time.time() + ttl_s,
    }
    return token


def _consume_pair_token(token: str) -> str | None:
    _prune_pair_tokens()
    data = _pair_tokens.pop(token, None)
    if not data:
        return None
    if data.get("expires_at", 0) < time.time():
        return None
    return data.get("email")


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
PROJECT_SETTINGS_FILE = "chat_settings.json"


def _safe_dataset_name(name: str) -> str:
    raw = (name or "dataset.csv").strip()
    raw = raw.replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    return cleaned or "dataset.csv"


def _normalize_target_col(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text or "@" in text:
        return None
    return text[:200]


def _project_settings_path(project_root: Path) -> Path:
    return project_root / "shared" / PROJECT_SETTINGS_FILE


def _read_project_settings(project_root: Path | None) -> dict:
    if not project_root:
        return {}
    path = _project_settings_path(project_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_project_settings(project_root: Path, settings: dict) -> dict:
    path = _project_settings_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def _require_auth(msg: dict) -> str:
    payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
    token = msg.get("auth_token") or payload.get("auth_token")
    return _require_auth_token(token)


def _require_auth_token(token: str) -> str:
    token = (token or "").strip()
    if not _tokens_loaded:
        _load_tokens()
    entry = _active_tokens.get(token)
    if not token or not entry:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if isinstance(entry, dict):
        if time.time() > entry.get("expires_at", 0):
            _active_tokens.pop(token, None)
            _save_tokens()
            raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
        return entry.get("email", "")
    return str(entry)


def _require_auth_from_request(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.query_params.get("auth_token", "").strip()
    return _require_auth_token(token)


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


def _resolve_authorized_project(payload: dict, email: str) -> Path:
    base = _ensure_platform_root().resolve()
    project_id = (payload.get("project_id") or "").strip()
    project_root_raw = (payload.get("project_root") or "").strip()

    if project_id:
        project_root = (base / project_id).resolve()
    elif project_root_raw:
        project_root = Path(project_root_raw).resolve()
    elif workspace.is_initialized and workspace.project_root:
        project_root = workspace.project_root.resolve()
    else:
        raise HTTPException(status_code=400, detail="Project not initialized.")

    if not project_root.exists() or not project_root.is_dir():
        raise HTTPException(status_code=404, detail="Project not found.")
    if project_root != base and base not in project_root.parents:
        raise HTTPException(status_code=400, detail="Invalid project path.")

    owner = _get_project_owner(project_root)
    if not owner:
        owner = _maybe_claim_legacy_project(project_root, email)
    if owner and owner != email.lower():
        raise HTTPException(status_code=403, detail="Forbidden.")
    return project_root


def _run_transparency_for_dataset_sync(dataset_path: Path, output_path: Path) -> None:
    code = (
        "import sys, pandas as pd\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from model_transparency import run_pipeline_from_df, infer_target_column\n"
        "dataset_path = sys.argv[1]\n"
        "df = pd.read_csv(dataset_path)\n"
        "target_col, _, _, _ = infer_target_column(df, verbose=False)\n"
        "y = df[target_col]\n"
        "task_type = 'classification' if y.nunique(dropna=True) <= 20 else 'regression'\n"
        "model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=120, random_state=42)\n"
        "print('DATASET_PATH:', dataset_path)\n"
        "print('DATASET_SHAPE:', df.shape)\n"
        "print('TARGET_COL:', target_col)\n"
        "print('TASK_TYPE:', task_type)\n"
        # This is a fast, background inspection report.  Keep the selected
        # RandomForest rather than auto-swapping to a 500-iteration CatBoost
        # run for wide/high-cardinality CSVs; detailed model training is the
        # ML Engineer agent's responsibility.
        "run_pipeline_from_df(model=model, df=df, target_col=target_col, task_type=task_type, test_size=0.2, scale=(task_type=='regression'), cv=3, n_walkthrough=2, disable_catboost_swap=True)\n"
    )
    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code, str(dataset_path)],
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


async def _process_dataset_background(dataset_path: Path, filename: str = "dataset.csv", task_id: str | None = None) -> None:
    if not workspace.is_initialized or not workspace.project_root:
        return
    print(f"[SERVER] ⏳ Agents analyzing dataset: {dataset_path} ...")
    shared_dir = workspace.project_root / "shared"
    out_path = shared_dir / "output.txt"
    rag_dir = shared_dir / "rag"

    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "data_analyst", "status": "BUSY"}})
    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "orchestrator", "status": "BUSY"}})
    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "ml_engineer", "status": "BUSY"}})

    target_col = ""
    conf = 0.0
    reason = "No target inferred"
    shape_str = "unknown shape"
    columns_list = []

    try:
        def _inspect_and_infer():
            import pandas as pd
            from model_transparency import infer_target_column
            if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(dataset_path)
            elif dataset_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(dataset_path)
            elif dataset_path.suffix.lower() == ".json":
                df = pd.read_json(dataset_path)
            elif dataset_path.suffix.lower() == ".tsv":
                df = pd.read_csv(dataset_path, sep="\t")
            else:
                df = pd.read_csv(dataset_path)
            t_col, t_conf, t_reason, _ = infer_target_column(df, verbose=False)
            return t_col, t_conf, t_reason, df.shape, list(df.columns)

        target_col, conf, reason, shape, columns_list = await asyncio.to_thread(_inspect_and_infer)
        shape_str = f"{shape[0]} rows × {shape[1]} columns"
        conf_display = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
        print(f"[SERVER] 🎯 Auto-detected Target Column: '{target_col}' (confidence: {conf_display}, reason: {reason})")
    except Exception as e:
        conf_display = str(conf) if conf else "N/A"
        print(f"[SERVER] ⚠️ Target column inference warning: {e}")

    try:
        await asyncio.to_thread(_run_transparency_for_dataset_sync, dataset_path, out_path)
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8", errors="replace")
            await asyncio.to_thread(build_hybrid_index_from_text, text, rag_dir)
            print(f"[SERVER] 📚 Built hybrid RAG index for dataset transparency summary.")
    except Exception as e:
        print(f"[SERVER] ⚠️ RAG index build warning: {e}")

    if target_col:
        try:
            settings = _read_project_settings(workspace.project_root)
            settings["target_col"] = target_col
            _write_project_settings(workspace.project_root, settings)
            print(f"[SERVER] 💾 Saved target column '{target_col}' to project settings.")
            await broadcast_to_gui({
                "type": "project_settings",
                "from": "server",
                "to": "gui",
                "content": "project_settings_updated",
                "target_col": target_col,
                "payload": {"target_col": target_col},
            })
        except Exception as e:
            print(f"[SERVER] ⚠️ Could not save project settings: {e}")

    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "data_analyst", "status": "IDLE"}})
    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "orchestrator", "status": "IDLE"}})
    await broadcast_to_gui({"type": "status", "payload": {"agent_id": "ml_engineer", "status": "IDLE"}})

    cols_preview = ", ".join([f"`{c}`" for c in columns_list[:10]])
    if len(columns_list) > 10:
        cols_preview += f" (+{len(columns_list) - 10} more)"
    if not 'conf_display' in locals():
        conf_display = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
    summary_msg = (
        f"📊 **Dataset Readiness & Inspection Complete**\n"
        f"- **File**: `{filename}`\n"
        f"- **Path**: `{dataset_path}`\n"
        f"- **Shape**: {shape_str}\n"
        f"- **Features ({len(columns_list)})**: {cols_preview or 'N/A'}\n"
        f"- **Assigned Target Column**: `{target_col or 'None'}` *(confidence: {conf_display})*\n"
        f"- **Selection Rationale**: {reason}\n\n"
        f"✅ *All agents (`Orchestrator`, `Data Analyst`, `ML Engineer`) indexed the dataset and are ready for tasks!*"
    )
    await broadcast_to_gui({
        "type": "message",
        "from": "data_analyst",
        "to": "orchestrator",
        "payload": {
            "chat_id": "team",
            "from": "data_analyst",
            "content": summary_msg,
            "tag": "STATUS",
        }
    })
    print(f"[SERVER] ✨ Dataset background processing complete for {filename}.")


# ─── Broadcast to all GUI clients ─────────────────────────────────────────────

async def broadcast_to_gui(message: dict) -> None:
    if not connected_clients:
        return
    user_scope = (message.get("user_email") or "").strip().lower()
    payload = json.dumps({**message, "timestamp": datetime.utcnow().isoformat()})
    dead = set()
    for ws in connected_clients:
        if user_scope:
            ws_user = _ws_users.get(ws, "").lower()
            if ws_user != user_scope:
                continue
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)
    for ws in dead:
        _ws_users.pop(ws, None)


# ─── Bus listeners (each runs exactly once) ───────────────────────────────────

def _detect_tag(content: str) -> str | None:
    if not content:
        return None
    u = content.upper()
    if any(k in u for k in ["ERROR", "FAIL", "EXCEPTION"]):
        return "ALERT"
    if any(k in u for k in ["SUCCESS", "SAVED", "COMPLETE", "COMPLETED", "DONE"]):
        return "SUCCESS"
    if any(k in u for k in ["REPORT", "FINDING", "SUGGESTION", "ANALYSIS"]):
        return "REPORT"
    return None


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
            "type":    "file_log",
            "from":    p.get("agent_id"),
            "to":      "gui",
            "content": p.get("content", ""),
            "tag":     "STATUS",
            "task_id": p.get("task_id"),
            "agent_id":  p.get("agent_id"),
            "filename":  p.get("filename"),
            "full_path": p.get("path"),
            "path":      p.get("path"),
            "extra": {
                "agent_id":  p.get("agent_id"),
                "filename":  p.get("filename"),
                "full_path": p.get("path"),
                "path":      p.get("path"),
            },
        })
        await broadcast_to_gui({
            "type":    "file_written",
            "from":    p.get("agent_id"),
            "to":      "gui",
            "content": p.get("content", ""),
            "tag":     "STATUS",
            "task_id": p.get("task_id"),
            "agent_id":  p.get("agent_id"),
            "filename":  p.get("filename"),
            "full_path": p.get("path"),
            "path":      p.get("path"),
            "extra": {
                "agent_id":  p.get("agent_id"),
                "filename":  p.get("filename"),
                "full_path": p.get("path"),
                "path":      p.get("path"),
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
            "user_email": _active_project_owner or "",
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
        settings = _read_project_settings(workspace.project_root)
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
                "target_col": _normalize_target_col(settings.get("target_col")),
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
                for f in files:
                    await websocket.send_text(json.dumps({
                        "type":      "file_log",
                        "from":      f["agent_id"],
                        "to":        "gui",
                        "content":   f"File: {f['filename']}",
                        "tag":       "STATUS",
                        "timestamp": datetime.utcnow().isoformat(),
                        "task_id":   None,
                        "agent_id":  f["agent_id"],
                        "filename":  f["filename"],
                        "full_path": f["full_path"],
                        "path":      f["full_path"],
                        "extra":     f,
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
                            "target_col": _normalize_target_col(_read_project_settings(project_root).get("target_col")),
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
                payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
                agent_id     = msg.get("agent_id") or payload.get("agent_id") or "shared"
                filename     = msg.get("filename") or payload.get("filename") or "output.txt"
                file_content = msg.get("content") if msg.get("content") is not None else payload.get("content", "")
                from_agent   = msg.get("from") or payload.get("from") or agent_id
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

            # ── Dataset upload from GUI (base64 over WebSocket) ───────────────
            if msg_type == "dataset_upload":
                try:
                    _require_auth(msg)
                except HTTPException:
                    continue
                if not workspace.is_initialized:
                    _ensure_platform_root()
                    workspace.new_project(f"chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

                payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
                filename = _safe_dataset_name(msg.get("filename") or payload.get("filename") or "dataset.csv")
                payload_b64 = msg.get("content_b64") or payload.get("content_b64") or ""
                task_id = msg.get("task_id") or payload.get("task_id") or str(uuid.uuid4())[:8]
                print(f"[SERVER] 📂 Receiving dataset upload: '{filename}' (Task ID: {task_id})")

                try:
                    blob = base64.b64decode(payload_b64, validate=True)
                    if len(blob) > 100_000_000:
                        raise ValueError("Dataset exceeds 100MB size limit.")
                    ext = Path(filename).suffix.lower()
                    if ext not in {".csv", ".tsv", ".json", ".parquet", ".xlsx"}:
                        raise ValueError("Unsupported dataset format.")
                    datasets_dir = workspace.project_root / "shared" / "datasets"
                    datasets_dir.mkdir(parents=True, exist_ok=True)
                    dataset_path = datasets_dir / filename
                    dataset_path.write_bytes(blob)
                    print(f"[SERVER] ✅ Dataset successfully written to: {dataset_path} ({len(blob)} bytes)")

                    await broadcast_to_gui({
                        "type": "dataset_uploaded",
                        "from": "server",
                        "to": "gui",
                        "content": "dataset_uploaded",
                        "tag": "STATUS",
                        "task_id": task_id,
                        "filename": filename,
                        "path": str(dataset_path),
                        "payload": {
                            "filename": filename,
                            "path": str(dataset_path),
                        },
                        "extra": {
                            "filename": filename,
                            "path": str(dataset_path),
                        },
                    })
                    print(f"[SERVER] 🚀 Launching background inspection and agent analysis for {filename}...")
                    asyncio.create_task(_process_dataset_background(dataset_path, filename, task_id))
                except Exception as e:
                    print(f"[SERVER] ❌ Error processing dataset upload: {e}")
                continue

            # ── User message → orchestrator ────────────────────────────────
            if msg_type == "set_project_target":
                try:
                    email = _require_auth(msg)
                    _ws_users[websocket] = email.lower()
                    _assert_project_owner(email)
                except HTTPException:
                    continue
                payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
                proj_path = (msg.get("project_root") or payload.get("project_root") or msg.get("worker_project_path") or payload.get("worker_project_path") or "").strip()
                if proj_path and Path(proj_path).exists():
                    if not workspace.is_initialized or str(workspace.project_root) != str(Path(proj_path).resolve()):
                        workspace.load_project(Path(proj_path))
                if not workspace.is_initialized or not workspace.project_root:
                    continue
                target_col = _normalize_target_col(msg.get("target_col") or payload.get("target_col"))
                try:
                    settings = _read_project_settings(workspace.project_root)
                    if target_col:
                        settings["target_col"] = target_col
                    else:
                        settings.pop("target_col", None)
                    _write_project_settings(workspace.project_root, settings)
                    await broadcast_to_gui({
                        "type": "project_settings",
                        "from": "server",
                        "to": "gui",
                        "content": "project_settings_updated",
                        "tag": None,
                        "task_id": msg.get("task_id"),
                        "user_email": email.lower(),
                        "extra": {
                            "project_root": str(workspace.project_root),
                            "target_col": _normalize_target_col(settings.get("target_col")),
                        },
                    })
                except Exception:
                    pass
                continue

            payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}
            content = (msg.get("content") or payload.get("content") or "").strip()
            to      = msg.get("to") or payload.get("chat_id") or payload.get("to") or "team"
            task_id = msg.get("task_id") or payload.get("task_id") or str(uuid.uuid4())[:8]

            if not content:
                continue

            try:
                email = _require_auth(msg)
                _ws_users[websocket] = email.lower()
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

            # Ensure workspace is loaded if project_root/worker_project_path is passed
            proj_path = (msg.get("project_root") or payload.get("project_root") or msg.get("worker_project_path") or payload.get("worker_project_path") or "").strip()
            if proj_path and Path(proj_path).exists():
                if not workspace.is_initialized or str(workspace.project_root) != str(Path(proj_path).resolve()):
                    workspace.load_project(Path(proj_path))

            # Guard against accidental duplicate sends from multiple GUI WS connections.
            target_col = _normalize_target_col(msg.get("target_col") or payload.get("target_col"))
            auth_token = msg.get("auth_token") or payload.get("auth_token")
            worker_project_path = proj_path or (str(workspace.project_root) if workspace.is_initialized else "")
            dedup_key = (to, content, target_col or "")
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
                extra={
                    "auth_token": auth_token,
                    "user_email": email,
                    "worker_project_path": worker_project_path,
                    "project_root": worker_project_path,
                    "target_col": target_col,
                },
            )


    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        _ws_users.pop(websocket, None)
        print(f"[SERVER] GUI disconnected. Remaining: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        _ws_users.pop(websocket, None)
        print(f"[SERVER] WebSocket error: {e}")


# ── Local worker WebSocket ──────────────────────────────────────────────────

@app.websocket("/worker")
async def worker_endpoint(websocket: WebSocket):
    await websocket.accept()
    worker_email = None
    try:
        raw = await websocket.receive_text()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.close(code=4000)
            return
        if msg.get("type") != "pair":
            await websocket.close(code=4001)
            return
        pair_token = (msg.get("token") or "").strip()
        worker_email = _consume_pair_token(pair_token)
        if not worker_email:
            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid or expired token"}))
            await websocket.close(code=4003)
            return
        _worker_sessions[worker_email] = websocket
        await websocket.send_text(json.dumps({"type": "paired", "email": worker_email}))
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result":
                job_id = msg.get("job_id")
                payload = msg.get("payload") or {}
                pending = _worker_pending.pop(job_id, None)
                if pending and not pending["future"].done():
                    pending["future"].set_result(payload)
            elif msg.get("type") == "log":
                content = msg.get("content") or ""
                if content:
                    await broadcast_to_gui({
                        "type": "team_message",
                        "from": "ml_engineer",
                        "to": "team",
                        "content": content,
                        "tag": _detect_tag(content),
                        "task_id": None,
                    })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if worker_email and _worker_sessions.get(worker_email) is websocket:
            _worker_sessions.pop(worker_email, None)
        to_fail = [jid for jid, meta in _worker_pending.items() if meta.get("email") == worker_email]
        for jid in to_fail:
            meta = _worker_pending.pop(jid, None)
            if meta and not meta["future"].done():
                meta["future"].set_result({
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "Local worker disconnected.",
                })

    c = content.lower()
    if any(k in c for k in ["✅", "complete", "done", "deployed", "approved", "passed", "ci passed"]):
        return "DONE"
    if any(k in c for k in ["🔴", "critical", "exploit", "blocked", "❌"]):
        return "ALERT"
    if any(k in c for k in ["found", "result", "report", "eda", "accuracy", "score", "metric"]):
        return "REPORT"
    return "STATUS"
