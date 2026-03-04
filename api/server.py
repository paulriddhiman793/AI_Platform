"""
api/server.py — FastAPI + WebSocket server

Bridges the Python agent system to the React GUI.

Flow:
  React GUI  →  WebSocket /ws  →  Orchestrator  →  Agents
  Agents     →  message bus    →  WebSocket /ws  →  React GUI

Message format (JSON) sent TO the GUI:
  {
    "type":      "team_message" | "p2p_message" | "agent_status" |
                 "file_written" | "project_info" | "error",
    "from":      agent_id,
    "to":        agent_id | "user",
    "content":   string,
    "tag":       "STATUS" | "REPORT" | "ALERT" | "DONE" | null,
    "timestamp": ISO string,
    "task_id":   string | null,
    "extra":     {} | null   (file path, project path, etc.)
  }

Message format (JSON) received FROM the GUI:
  {
    "type":    "user_message" | "direct_message" | "ping",
    "content": string,
    "to":      agent_id | "team",
    "chat_id": string
  }

Usage:
  python -m api.main   (starts everything)
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.message_bus import bus, send_to_agent
from tools.workspace import workspace


app = FastAPI(title="AI Engineering Platform")

# Allow React dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track all connected WebSocket clients
connected_clients: Set[WebSocket] = set()


# ─── Broadcast to all connected GUI clients ───────────────────────────────────

async def broadcast_to_gui(message: dict) -> None:
    """Send a JSON message to every connected React client."""
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


# ─── Agent message bus listeners ─────────────────────────────────────────────

async def listen_orchestrator_inbox() -> None:
    """
    Forward all agent reports (orchestrator.inbox) to the GUI as team messages.
    """
    q = bus.subscribe("orchestrator.inbox")
    while True:
        envelope = await q.get()
        payload = envelope["payload"]
        from_agent = payload.get("from", "unknown")
        content    = payload.get("content", "")
        task_id    = payload.get("task_id")

        # Detect tag from content keywords
        tag = _detect_tag(content)

        await broadcast_to_gui({
            "type":    "team_message",
            "from":    from_agent,
            "to":      "team",
            "content": content,
            "tag":     tag,
            "task_id": task_id,
        })


async def listen_p2p() -> None:
    """
    Forward all p2p agent-to-agent messages to the GUI as activity feed entries.
    We subscribe to a special "p2p.monitor" channel that agents publish to.
    """
    q = bus.subscribe("p2p.monitor")
    while True:
        envelope = await q.get()
        payload = envelope["payload"]
        await broadcast_to_gui({
            "type":    "p2p_message",
            "from":    payload.get("from"),
            "to":      payload.get("to"),
            "content": payload.get("content", ""),
            "tag":     None,
            "task_id": payload.get("task_id"),
        })


async def listen_user_output() -> None:
    """Forward direct user-targeted messages to the GUI."""
    q = bus.subscribe("user.output")
    while True:
        envelope = await q.get()
        payload = envelope["payload"]
        await broadcast_to_gui({
            "type":    "team_message",
            "from":    payload.get("from"),
            "to":      "user",
            "content": payload.get("content", ""),
            "tag":     "DONE",
            "task_id": payload.get("task_id"),
        })


async def listen_file_events() -> None:
    """Forward workspace file-write events to the GUI."""
    q = bus.subscribe("workspace.files")
    while True:
        envelope = await q.get()
        payload = envelope["payload"]
        await broadcast_to_gui({
            "type":    "file_written",
            "from":    payload.get("agent_id"),
            "to":      "user",
            "content": f"File written: {payload.get('path')}",
            "tag":     "STATUS",
            "task_id": payload.get("task_id"),
            "extra":   {
                "filename": payload.get("filename"),
                "agent_id": payload.get("agent_id"),
                "full_path": payload.get("path"),
            },
        })


async def listen_agent_status() -> None:
    """Forward agent status changes (idle/working/error) to the GUI."""
    q = bus.subscribe("agent.status")
    while True:
        envelope = await q.get()
        payload = envelope["payload"]
        await broadcast_to_gui({
            "type":    "agent_status",
            "from":    payload.get("agent_id"),
            "to":      "gui",
            "content": payload.get("status"),
            "tag":     None,
            "task_id": None,
        })


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[SERVER] GUI connected. Total clients: {len(connected_clients)}")

    # Send project info immediately on connect
    if workspace.is_initialized:
        await websocket.send_text(json.dumps({
            "type":      "project_info",
            "from":      "server",
            "to":        "gui",
            "content":   f"Connected to project: {workspace.project_name}",
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

            msg_type = msg.get("type", "user_message")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong", "from": "server", "to": "gui",
                    "content": "pong", "tag": None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_id": None,
                }))
                continue

            # User sent a message — route to orchestrator or specific agent
            content = msg.get("content", "").strip()
            to      = msg.get("to", "team")
            task_id = msg.get("task_id") or str(uuid.uuid4())[:8]

            if not content:
                continue

            if to == "team":
                await send_to_agent(
                    from_agent="user",
                    to_agent="orchestrator",
                    content=content,
                    task_id=task_id,
                )
            else:
                # Direct message to a specific agent
                await send_to_agent(
                    from_agent="user",
                    to_agent=to,
                    content=content,
                    task_id=task_id,
                )

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"[SERVER] GUI disconnected. Remaining: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        print(f"[SERVER] WebSocket error: {e}")


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project": workspace.project_name,
        "project_root": str(workspace.project_root) if workspace.project_root else None,
        "clients": len(connected_clients),
    }

@app.get("/files")
async def list_files():
    """Return all files written so far in the current project."""
    if not workspace.is_initialized:
        return {"files": []}
    return {"files": workspace.list_files(), "root": str(workspace.project_root)}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _detect_tag(content: str) -> str:
    """Infer a display tag from message content."""
    c = content.lower()
    if any(kw in c for kw in ["✅", "complete", "done", "deployed", "approved", "passed", "ci passed"]):
        return "DONE"
    if any(kw in c for kw in ["🔴", "critical", "exploit", "blocked", "❌"]):
        return "ALERT"
    if any(kw in c for kw in ["found", "result", "report", "eda", "accuracy", "score", "metric"]):
        return "REPORT"
    return "STATUS"


# ─── Startup: launch all bus listeners ───────────────────────────────────────

@app.on_event("startup")
async def start_listeners():
    """Start all message bus listeners when FastAPI starts."""
    asyncio.create_task(listen_orchestrator_inbox())
    asyncio.create_task(listen_p2p())
    asyncio.create_task(listen_user_output())
    asyncio.create_task(listen_file_events())
    asyncio.create_task(listen_agent_status())
    print("[SERVER] All bus listeners started.")