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
from datetime import datetime
from pathlib import Path
from typing import Set
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.message_bus import bus, send_to_agent
from tools.rag_store import build_hybrid_index_from_text
from tools.workspace import workspace


app = FastAPI(title="AI Engineering Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: Set[WebSocket] = set()
_recent_user_messages: dict[tuple[str, str], float] = {}

# ── Guard: listeners only ever start once ─────────────────────────────────────
_listeners_started = False


def _safe_dataset_name(name: str) -> str:
    raw = (name or "dataset.csv").strip()
    raw = raw.replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    return cleaned or "dataset.csv"


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

            # ── File write from frontend ───────────────────────────────────
            if msg_type == "file_write":
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
                if not workspace.is_initialized:
                    continue

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
                except Exception:
                    pass
                continue

            # ── User message → orchestrator ────────────────────────────────
            content = msg.get("content", "").strip()
            to      = msg.get("to", "team")
            task_id = msg.get("task_id") or str(uuid.uuid4())[:8]

            if not content:
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
