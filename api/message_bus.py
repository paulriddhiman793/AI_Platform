"""
api/message_bus.py - Agent Message Bus

Notes:
- p2p.monitor is a GUI mirror channel, not an agent inbox
- A single bus instance is used process-wide
"""
import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Optional


class MockMessageBus:
    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._log: list[dict] = []

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(q)
        return q

    async def publish(self, channel: str, message: dict) -> None:
        envelope = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": message,
        }
        self._log.append(envelope)

        # p2p.monitor is a mirror copy; skipping it avoids duplicate-looking CLI logs.
        if channel != "p2p.monitor":
            preview = str(message.get("content", ""))[:60]
            print(f"[BUS] {message.get('from', '?')} -> {message.get('to', channel)} | {preview}")

        for q in self._subscribers.get(channel, []):
            await q.put(envelope)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, []))

    def get_log(self) -> list[dict]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()


bus = MockMessageBus()
_recent_p2p: dict[tuple[str, str, str], float] = {}


def _dedup_p2p(from_agent: str, to_agent: str, content: str, window_s: float = 1.5) -> bool:
    """Return True if duplicate within window; otherwise record and return False."""
    key = (from_agent, to_agent, content)
    now_ts = datetime.utcnow().timestamp()
    last_ts = _recent_p2p.get(key, 0.0)
    if now_ts - last_ts < window_s:
        return True
    _recent_p2p[key] = now_ts
    if len(_recent_p2p) > 500:
        _recent_p2p.pop(next(iter(_recent_p2p)))
    return False


async def send_to_agent(
    from_agent: str,
    to_agent: str,
    content: str,
    task_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Send a p2p message to an agent inbox and copy it to p2p.monitor for GUI activity feed.
    """
    if _dedup_p2p(from_agent, to_agent, content):
        return
    msg = {
        "from": from_agent,
        "to": to_agent,
        "content": content,
        "task_id": task_id,
        "type": "p2p",
    }
    if extra:
        msg.update(extra)
    await bus.publish(f"agent.{to_agent}", msg)
    await bus.publish("p2p.monitor", msg)


async def broadcast(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    """Broadcast a result to orchestrator inbox (shown in team/user flow by server)."""
    await bus.publish("orchestrator.inbox", {
        "from": from_agent,
        "to": "orchestrator",
        "content": content,
        "task_id": task_id,
        "type": "broadcast",
    })


async def send_to_user(
    from_agent: str,
    content: str,
    task_id: Optional[str] = None,
) -> None:
    await bus.publish("user.output", {
        "from": from_agent,
        "to": "user",
        "content": content,
        "task_id": task_id,
        "type": "user_message",
    })


async def publish_status(agent_id: str, status: str) -> None:
    await bus.publish("agent.status", {
        "agent_id": agent_id,
        "status": status,
        "from": agent_id,
        "to": "gui",
    })


async def publish_file_event(
    agent_id: str,
    filename: str,
    full_path: str,
    task_id: Optional[str] = None,
) -> None:
    await bus.publish("workspace.files", {
        "agent_id": agent_id,
        "filename": filename,
        "path": full_path,
        "task_id": task_id,
        "from": agent_id,
        "to": "gui",
        "content": f"File written: {filename}",
    })
